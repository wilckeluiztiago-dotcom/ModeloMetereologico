import numpy as np

"""
MÓDULO DE SENSORIAMENTO REMOTO: GEORREFERENCIAMENTO
===================================================

Atribui coordenadas de latitude/longitude a pixels de imagem.
Conversão (Linha, Coluna) -> (Lat, Lon) usando Transformada Afim.

AUTOR: Luiz Tiago Wilcke
DATA: 2024
"""

class Georreferenciador:
    def __init__(self, top_left_lat, top_left_lon, pixel_size_deg):
        self.tl_lat = top_left_lat
        self.tl_lon = top_left_lon
        self.px_size = pixel_size_deg
        
        # Transformada Afim simples (Norte para cima)
        # x_geo = GT[0] + x_pix * GT[1] + y_pix * GT[2]
        # y_geo = GT[3] + x_pix * GT[4] + y_pix * GT[5]
        self.geo_transform = [
            top_left_lon, pixel_size_deg, 0,
            top_left_lat, 0, -pixel_size_deg
        ]
        
    def pixel_para_latlon(self, row, col):
        """Retorna (Lat, Lon) do centro do pixel."""
        # Centro do pixel = índice + 0.5
        c = col + 0.5
        r = row + 0.5
        
        lon = self.geo_transform[0] + c * self.geo_transform[1] + r * self.geo_transform[2]
        lat = self.geo_transform[3] + c * self.geo_transform[4] + r * self.geo_transform[5]
        
        return lat, lon

    def latlon_para_pixel(self, lat, lon):
        """Retorna (Row, Col)."""
        # Invertendo a lógica simples (assumindo sem rotação)
        # lon = lon0 + col * dlon
        col = (lon - self.tl_lon) / self.px_size
        
        # lat = lat0 - row * dlat
        row = (self.tl_lat - lat) / self.px_size
        
        return int(row), int(col)

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testando Georreferenciamento...")
    
    geo = Georreferenciador(top_left_lat=-30.0, top_left_lon=-55.0, pixel_size_deg=0.01)
    
    r, c = 100, 100
    lat, lon = geo.pixel_para_latlon(r, c)
    print(f"Pixel ({r}, {c}) -> Lat: {lat:.4f}, Lon: {lon:.4f}")
    
    r_back, c_back = geo.latlon_para_pixel(lat, lon)
    print(f"Recuperado: ({r_back}, {c_back})")
