"""
SOAP 協議工具模組

為 JHTDB Web Services API 提供 SOAP 請求構建和響應解析功能。

Classes:
    SOAPRequest: SOAP 請求構建器，包含重試邏輯
    SOAPResponseParser: SOAP 響應解析器，處理二進制和 XML 數據

Author: PINNs-MVP Team
Date: 2025-12-15
"""

import urllib.request
import urllib.error
import base64
import time
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class SOAPRequest:
    """SOAP 請求構建器與發送器
    
    處理 JHTDB Web Services API 的 SOAP 協議通信：
    - 構建 SOAP XML 請求
    - 發送 HTTP 請求與重試邏輯
    - 提取 Base64 編碼的二進制響應
    
    Attributes:
        base_url: JHTDB SOAP 服務端點
        auth_token: 認證令牌
        timeout: 請求超時時間（秒）
        max_retry: 最大重試次數
    """
    
    def __init__(self, 
                 base_url: str,
                 auth_token: str,
                 timeout: int = 300,
                 max_retry: int = 3):
        """
        Args:
            base_url: JHTDB SOAP 服務 URL (e.g., https://turbulence.pha.jhu.edu/service/turbulence.asmx)
            auth_token: JHTDB 認證令牌
            timeout: 請求超時時間（秒）
            max_retry: 最大重試次數
        """
        self.base_url = base_url
        self.auth_token = auth_token
        self.timeout = timeout
        self.max_retry = max_retry
    
    def build_get_any_cutout_request(self, 
                                     dataset: str,
                                     field: str,
                                     start: List[int],
                                     end: List[int],
                                     timestep: int) -> str:
        """構建 GetAnyCutoutWeb SOAP 請求
        
        Args:
            dataset: 資料集名稱 (e.g., 'channel', 'isotropic1024coarse')
            field: 場類型 ('velocity', 'pressure')
            start: 起始網格索引 [x, y, z] (1-based)
            end: 結束網格索引 [x, y, z] (1-based)
            timestep: 時間步（整數）
        
        Returns:
            SOAP XML 請求字串
        """
        return f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
               xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
               xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <GetAnyCutoutWeb xmlns="http://turbulence.pha.jhu.edu/">
      <authToken>{self.auth_token}</authToken>
      <dataset>{dataset}</dataset>
      <field>{field}</field>
      <T>{timestep}</T>
      <x_start>{start[0]}</x_start>
      <y_start>{start[1]}</y_start>
      <z_start>{start[2]}</z_start>
      <x_end>{end[0]}</x_end>
      <y_end>{end[1]}</y_end>
      <z_end>{end[2]}</z_end>
      <x_step>1</x_step>
      <y_step>1</y_step>
      <z_step>1</z_step>
      <filter_width>1</filter_width>
      <addr></addr>
    </GetAnyCutoutWeb>
  </soap:Body>
</soap:Envelope>"""
    
    def build_get_velocity_request(self,
                                   dataset: str,
                                   points: List[List[float]],
                                   timestep: float) -> str:
        """構建 GetVelocity SOAP 請求（散點插值）
        
        Args:
            dataset: 資料集名稱
            points: 查詢點列表 [[x, y, z], ...] (物理座標)
            timestep: 時間步（浮點數）
        
        Returns:
            SOAP XML 請求字串
        """
        # 構建點的 XML 格式
        points_xml = ""
        for point in points:
            points_xml += f"""
        <Point3>
          <x>{point[0]}</x>
          <y>{point[1]}</y>
          <z>{point[2]}</z>
        </Point3>"""
        
        return f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
               xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
               xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <GetVelocity xmlns="http://turbulence.pha.jhu.edu/">
      <authToken>{self.auth_token}</authToken>
      <dataset>{dataset}</dataset>
      <time>{float(timestep)}</time>
      <spatialInterpolation>Lag4</spatialInterpolation>
      <temporalInterpolation>None</temporalInterpolation>
      <points>{points_xml}
      </points>
    </GetVelocity>
  </soap:Body>
</soap:Envelope>"""
    
    def build_get_pressure_request(self,
                                   dataset: str,
                                   points: List[List[float]],
                                   timestep: float) -> str:
        """構建 GetPressure SOAP 請求（散點插值）
        
        Args:
            dataset: 資料集名稱
            points: 查詢點列表 [[x, y, z], ...] (物理座標)
            timestep: 時間步（浮點數）
        
        Returns:
            SOAP XML 請求字串
        """
        # 編碼點為 Base64 二進制格式
        points_binary = self._encode_points(points)
        
        return f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
               xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
               xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <GetPressure xmlns="http://turbulence.pha.jhu.edu/">
      <authToken>{self.auth_token}</authToken>
      <dataset>{dataset}</dataset>
      <time>{timestep}</time>
      <spatialInterpolation>6</spatialInterpolation>
      <temporalInterpolation>0</temporalInterpolation>
       <points>{points_binary}</points>
     </GetPressure>
   </soap:Body>
</soap:Envelope>"""
    
    def send_request(self, soap_xml: str, api_method: str) -> bytes:
        """發送 SOAP 請求並返回二進制響應
        
        包含重試邏輯與錯誤處理。
        
        Args:
            soap_xml: SOAP XML 請求字串
            api_method: API 方法名稱 (e.g., 'GetAnyCutoutWeb')
        
        Returns:
            從響應中提取的二進制數據
        
        Raises:
            JHTDBError: 所有重試失敗或響應解析錯誤
        """
        # 設置 HTTP 頭
        headers = {
            'Content-Type': 'text/xml; charset=utf-8',
            'SOAPAction': f'http://turbulence.pha.jhu.edu/{api_method}',
            'User-Agent': 'Python JHTDB Client'
        }
        
        # 編碼請求
        request_data = soap_xml.encode('utf-8')
        
        # 創建請求
        req = urllib.request.Request(
            self.base_url,
            data=request_data,
            headers=headers
        )
        
        # 重試邏輯
        for attempt in range(self.max_retry):
            try:
                logger.debug(f"發送 SOAP 請求（嘗試 {attempt + 1}/{self.max_retry}）")
                
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    if response.status == 200:
                        response_data = response.read()
                        logger.debug(f"請求成功，響應大小: {len(response_data)} bytes")
                        return self.extract_binary_data(response_data)
                    else:
                        from pinnx.dataio.jhtdb_client import JHTDBError
                        raise JHTDBError(f"HTTP 錯誤: {response.status}")
            
            except urllib.error.URLError as e:
                logger.warning(f"請求失敗（嘗試 {attempt + 1}）: {e}")
                if attempt == self.max_retry - 1:
                    from pinnx.dataio.jhtdb_client import JHTDBError
                    raise JHTDBError(f"所有重試均失敗: {e}")
                time.sleep(2 ** attempt)  # 指數退避
            
            except Exception as e:
                logger.error(f"未預期的錯誤: {e}")
                if attempt == self.max_retry - 1:
                    from pinnx.dataio.jhtdb_client import JHTDBError
                    raise JHTDBError(f"請求處理失敗: {e}")
                time.sleep(2 ** attempt)
        
        # 如果所有重試都失敗，拋出錯誤
        from pinnx.dataio.jhtdb_client import JHTDBError
        raise JHTDBError("所有連接嘗試均失敗")
    
    def extract_binary_data(self, response_data: bytes) -> bytes:
        """從 SOAP 響應中提取 Base64 編碼的二進制數據
        
        Args:
            response_data: SOAP XML 響應（bytes）
        
        Returns:
            解碼後的二進制數據
        
        Raises:
            JHTDBError: XML 解析失敗或找不到數據
        """
        try:
            # 解析 XML 響應
            response_str = response_data.decode('utf-8')
            root = ET.fromstring(response_str)
            
            # 尋找包含 Base64 數據的元素
            namespaces = {
                'soap': 'http://schemas.xmlsoap.org/soap/envelope/',
                'jhtdb': 'http://turbulence.pha.jhu.edu/'
            }
            
            # 尋找結果元素（可能在不同位置）
            result_elements = root.findall('.//soap:Body/*/*', namespaces)
            
            if not result_elements:
                # 嘗試沒有命名空間的查找
                result_elements = root.findall('.//Body/*/*')
            
            if result_elements:
                base64_data = result_elements[0].text
                if base64_data:
                    return base64.b64decode(base64_data)
                else:
                    logger.error("Base64 數據為空")
                    from pinnx.dataio.jhtdb_client import JHTDBError
                    raise JHTDBError("響應中的數據為空")
            
            # 如果仍然沒有找到，記錄響應內容用於調試
            logger.error("無法從 SOAP 響應中提取數據")
            logger.debug(f"響應內容（前1000字符）: {response_str[:1000]}")
            from pinnx.dataio.jhtdb_client import JHTDBError
            raise JHTDBError("無法解析 SOAP 響應")
        
        except ET.ParseError as e:
            logger.error(f"XML 解析失敗: {e}")
            from pinnx.dataio.jhtdb_client import JHTDBError
            raise JHTDBError(f"響應格式錯誤: {e}")
        except Exception as e:
            logger.error(f"數據提取失敗: {e}")
            from pinnx.dataio.jhtdb_client import JHTDBError
            raise JHTDBError(f"響應處理失敗: {e}")
    
    @staticmethod
    def _encode_points(points: List[List[float]]) -> str:
        """將點座標編碼為 Base64 二進制格式
        
        JHTDB 期望的格式：float32 陣列，每個點 3 個座標 (x, y, z)
        
        Args:
            points: 點列表 [[x, y, z], ...]
        
        Returns:
            Base64 編碼字串
        """
        points_array = np.array(points, dtype=np.float32)
        binary_data = points_array.tobytes()
        return base64.b64encode(binary_data).decode('ascii')


class SOAPResponseParser:
    """SOAP 響應解析器
    
    解析 JHTDB Web Services API 返回的數據：
    - 速度/壓力場 cutout（二進制格式）
    - 散點插值結果（XML 或二進制格式）
    """
    
    @staticmethod
    def parse_velocity_cutout(binary_data: bytes, width: List[int]) -> np.ndarray:
        """解析速度場 cutout 響應
        
        JHTDB velocity 數據格式: float32, shape=[width[0], width[1], width[2], 3]
        
        Args:
            binary_data: 二進制響應數據
            width: 網格尺寸 [nx, ny, nz]
        
        Returns:
            速度場陣列，shape=[width[0], width[1], width[2], 3]
        """
        expected_size = width[0] * width[1] * width[2] * 3 * 4  # 4 bytes per float32
        
        if len(binary_data) != expected_size:
            logger.warning(f"數據大小不符：期望 {expected_size}, 實際 {len(binary_data)}")
        
        # 解析為 float32 陣列
        data_array = np.frombuffer(binary_data, dtype=np.float32)
        
        # 重塑為 [width[0], width[1], width[2], 3]
        return data_array.reshape(width[0], width[1], width[2], 3)
    
    @staticmethod
    def parse_pressure_cutout(binary_data: bytes, width: List[int]) -> np.ndarray:
        """解析壓力場 cutout 響應
        
        JHTDB pressure 數據格式: float32, shape=[width[0], width[1], width[2]]
        
        Args:
            binary_data: 二進制響應數據
            width: 網格尺寸 [nx, ny, nz]
        
        Returns:
            壓力場陣列，shape=[width[0], width[1], width[2]]
        """
        expected_size = width[0] * width[1] * width[2] * 4  # 4 bytes per float32
        
        if len(binary_data) != expected_size:
            logger.warning(f"數據大小不符：期望 {expected_size}, 實際 {len(binary_data)}")
        
        # 解析為 float32 陣列
        data_array = np.frombuffer(binary_data, dtype=np.float32)
        
        # 重塑為 [width[0], width[1], width[2]]
        return data_array.reshape(width[0], width[1], width[2])
    
    @staticmethod
    def parse_velocity_points(response_data: bytes, n_points: int) -> np.ndarray:
        """解析 GetVelocity 散點插值響應
        
        支持兩種格式：
        1. XML 格式：包含 Vector3 元素
        2. 二進制格式：float32 陣列（回退選項）
        
        Args:
            response_data: SOAP 響應數據（可能是 XML 或二進制）
            n_points: 查詢點數量
        
        Returns:
            速度場陣列，shape=[n_points, 3]
        """
        try:
            # 首先嘗試 XML 解析
            response_str = response_data.decode('utf-8')
            root = ET.fromstring(response_str)
            
            # 尋找 GetVelocityResult 元素
            namespaces = {
                'soap': 'http://schemas.xmlsoap.org/soap/envelope/',
                'jhtdb': 'http://turbulence.pha.jhu.edu/'
            }
            
            result_elem = root.find('.//jhtdb:GetVelocityResult', namespaces)
            if result_elem is None:
                # 嘗試不使用命名空間
                result_elem = root.find('.//GetVelocityResult')
            
            if result_elem is None:
                logger.error("無法找到 GetVelocityResult 元素")
                logger.debug(f"響應內容: {response_str[:1000]}")
                from pinnx.dataio.jhtdb_client import JHTDBError
                raise JHTDBError("響應格式錯誤：找不到結果元素")
            
            # 解析 Vector3 元素
            vectors = []
            vector_elems = result_elem.findall('.//Vector3') or result_elem.findall('.//jhtdb:Vector3', namespaces)
            
            for vector_elem in vector_elems:
                x = float(vector_elem.find('x').text or vector_elem.find('jhtdb:x', namespaces).text)
                y = float(vector_elem.find('y').text or vector_elem.find('jhtdb:y', namespaces).text)
                z = float(vector_elem.find('z').text or vector_elem.find('jhtdb:z', namespaces).text)
                vectors.append([x, y, z])
            
            if len(vectors) != n_points:
                logger.warning(f"返回的點數不符：期望 {n_points}, 實際 {len(vectors)}")
            
            return np.array(vectors, dtype=np.float32)
        
        except Exception as e:
            logger.error(f"解析 Vector3 響應失敗: {e}")
            # 回退到二進制解析
            try:
                # 假設 response_data 已經是二進制格式
                expected_size = n_points * 3 * 4  # 4 bytes per float32
                
                if len(response_data) != expected_size:
                    logger.warning(f"數據大小不符：期望 {expected_size}, 實際 {len(response_data)}")
                
                data_array = np.frombuffer(response_data, dtype=np.float32)
                return data_array.reshape(n_points, 3)
            
            except Exception as e2:
                logger.error(f"二進制回退解析也失敗: {e2}")
                from pinnx.dataio.jhtdb_client import JHTDBError
                raise JHTDBError(f"無法解析響應數據: {e}")
    
    @staticmethod
    def parse_pressure_points(binary_data: bytes, n_points: int) -> np.ndarray:
        """解析壓力場散點插值響應
        
        JHTDB pressure points 數據格式: float32, shape=[n_points]
        
        Args:
            binary_data: 二進制響應數據
            n_points: 查詢點數量
        
        Returns:
            壓力值陣列，shape=[n_points]
        """
        expected_size = n_points * 4  # 4 bytes per float32
        
        if len(binary_data) != expected_size:
            logger.warning(f"數據大小不符：期望 {expected_size}, 實際 {len(binary_data)}")
        
        # 解析為 float32 陣列
        data_array = np.frombuffer(binary_data, dtype=np.float32)
        
        # 返回一維陣列
        return data_array


# 測試代碼
if __name__ == "__main__":
    print("=== SOAP Utils Module Test ===\n")
    
    # Test 1: SOAPRequest 初始化
    print("Test 1: SOAPRequest Initialization")
    soap_request = SOAPRequest(
        base_url="https://turbulence.pha.jhu.edu/service/turbulence.asmx",
        auth_token="test_token_12345",
        timeout=300,
        max_retry=3
    )
    print(f"✅ SOAPRequest created: base_url={soap_request.base_url}, timeout={soap_request.timeout}")
    
    # Test 2: 構建 GetAnyCutoutWeb 請求
    print("\nTest 2: Build GetAnyCutoutWeb Request")
    cutout_xml = soap_request.build_get_any_cutout_request(
        dataset="channel",
        field="velocity",
        start=[1, 1, 1],
        end=[128, 32, 96],
        timestep=1
    )
    assert "<GetAnyCutoutWeb" in cutout_xml
    assert "<authToken>test_token_12345</authToken>" in cutout_xml
    assert "<field>velocity</field>" in cutout_xml
    print(f"✅ GetAnyCutoutWeb XML generated ({len(cutout_xml)} chars)")
    
    # Test 3: 構建 GetVelocity 請求
    print("\nTest 3: Build GetVelocity Request")
    points = [[3.14, 0.5, 1.57], [6.28, 0.8, 3.14]]
    velocity_xml = soap_request.build_get_velocity_request(
        dataset="channel",
        points=points,
        timestep=1.0
    )
    assert "<GetVelocity" in velocity_xml
    assert "<Point3>" in velocity_xml
    assert "<x>3.14</x>" in velocity_xml
    print(f"✅ GetVelocity XML generated ({len(velocity_xml)} chars)")
    
    # Test 4: 構建 GetPressure 請求
    print("\nTest 4: Build GetPressure Request")
    pressure_xml = soap_request.build_get_pressure_request(
        dataset="channel",
        points=points,
        timestep=1.0
    )
    assert "<GetPressure" in pressure_xml
    assert "<points>" in pressure_xml
    print(f"✅ GetPressure XML generated ({len(pressure_xml)} chars)")
    
    # Test 5: 編碼點座標
    print("\nTest 5: Encode Points to Base64")
    encoded = SOAPRequest._encode_points(points)
    decoded = base64.b64decode(encoded)
    decoded_points = np.frombuffer(decoded, dtype=np.float32).reshape(-1, 3)
    np.testing.assert_array_almost_equal(decoded_points, points)
    print(f"✅ Points encoded/decoded correctly: {decoded_points.shape}")
    
    # Test 6: SOAPResponseParser - velocity cutout
    print("\nTest 6: Parse Velocity Cutout")
    width = [16, 16, 16]
    test_velocity_data = np.random.randn(width[0], width[1], width[2], 3).astype(np.float32)
    binary_data = test_velocity_data.tobytes()
    
    parsed = SOAPResponseParser.parse_velocity_cutout(binary_data, width)
    np.testing.assert_array_equal(parsed, test_velocity_data)
    print(f"✅ Velocity cutout parsed: shape={parsed.shape}")
    
    # Test 7: SOAPResponseParser - pressure cutout
    print("\nTest 7: Parse Pressure Cutout")
    test_pressure_data = np.random.randn(width[0], width[1], width[2]).astype(np.float32)
    binary_data = test_pressure_data.tobytes()
    
    parsed = SOAPResponseParser.parse_pressure_cutout(binary_data, width)
    np.testing.assert_array_equal(parsed, test_pressure_data)
    print(f"✅ Pressure cutout parsed: shape={parsed.shape}")
    
    # Test 8: SOAPResponseParser - pressure points
    print("\nTest 8: Parse Pressure Points")
    n_points = 10
    test_pressure_points = np.random.randn(n_points).astype(np.float32)
    binary_data = test_pressure_points.tobytes()
    
    parsed = SOAPResponseParser.parse_pressure_points(binary_data, n_points)
    np.testing.assert_array_equal(parsed, test_pressure_points)
    print(f"✅ Pressure points parsed: shape={parsed.shape}")
    
    print("\n" + "="*50)
    print("✅ All 8 tests passed!")
    print("="*50)
