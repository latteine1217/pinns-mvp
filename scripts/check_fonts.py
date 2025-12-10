import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

def check_fonts():
    print("Searching for Chinese fonts...")
    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    
    chinese_fonts = []
    target_fonts = ['Arial Unicode MS', 'PingFang', 'Heiti', 'STHeiti', 'SimHei', 'Microsoft JhengHei', 'WenQuanYi']
    
    for font_path in font_list:
        try:
            prop = fm.FontProperties(fname=font_path)
            name = prop.get_name()
            for target in target_fonts:
                if target.lower() in name.lower() or target.lower() in font_path.lower():
                    chinese_fonts.append((name, font_path))
                    break
        except:
            continue
            
    if chinese_fonts:
        print(f"Found {len(chinese_fonts)} potential Chinese fonts:")
        for name, path in chinese_fonts:
            print(f"  - Name: {name}, Path: {path}")
    else:
        print("No common Chinese fonts found in system paths.")
        
    print("\nCurrent rcParams['font.sans-serif']:")
    print(plt.rcParams['font.sans-serif'])

if __name__ == "__main__":
    check_fonts()
