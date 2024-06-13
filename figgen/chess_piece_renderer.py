import svgwrite


def create_svg_with_unicode(filename, text, font_size="40px"):
    dwg = svgwrite.Drawing(filename, profile="tiny")
    dwg.add(dwg.text(text, insert=(10, 50), font_size=font_size, font_family="Arial"))
    dwg.save()


# Unicode character for the chess rook
chess_rook_unicode = "\u265C"

# Create SVG file
create_svg_with_unicode("chess_rook.svg", chess_rook_unicode)
