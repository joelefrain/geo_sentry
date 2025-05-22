import matplotlib.font_manager as fm
for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    if 'arial' in font.lower():
        print(fm.FontProperties(fname=font).get_name(), '->', font)
