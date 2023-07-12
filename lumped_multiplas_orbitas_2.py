"""
Terceiro modelo lumped para discretização do satélite.

Será escrito as classes para cada componentes do satélite, ao todo serão 6:
    - Subtrato do painel;
    - Estrutura do painel;
    - Estrutura do cubesat;
    - PCD's;
    - Parafusos e pressão de conexão;
    - Baterias.

Discretizando os compoenetes será possível calcular a conexão entre eles em radiação e condução

Descrição do cubesat:
    - Tamanho 1U;
    - 4 paineis solares;
    - 4 bases de cobre
    - 4 baes do painel
    - 2 substratos por paienl;
    - 4 PCB's;
    - 5 packs de 4 parafusos;
    - 12 parafusos de 20 mm;
    - 8 parafusos de 14 mm;
    - duas tampas de alumínio no topo;
    - 4 Estruturas verticais;
    - 8 estruturas horizontais.
    - Total: 54 nós.

Dados dos materiais obtidos do livro Spacecraft Thermal Control Handkbook vol. 1
"""

from gebhart_fator import *
from datetime import datetime

inicio = datetime.now()

import numpy as np
from tqdm import tqdm
import pandas as pd
import gc
import os, sys


# Criar as classes de cada componente

class Substrato:

    def __init__(self, id, nome, vol, area, e, a, rho, k, cp, **kwargs):
        self.id = id
        self.nome = nome
        self.vol = vol
        self.area = area
        self.e = e
        self.a = a
        self.rho = rho
        self.k = k
        self.cp = cp
        self.kwargs = kwargs

class BasePainel:
    def __init__(self, id, nome, vol, n, e, a, rho, k, cp,area, **kwargs ):
        self.id = id
        self.nome = nome
        self.vol = vol
        self.n = n
        self.e = e
        self.a = a
        self.rho = rho
        self.k = k
        self.cp = cp
        self.area = area
        self.kwargs = kwargs

class BaseCobre:
    def __init__(self, id, nome, vol, n, e, a, rho, k, cp,area, **kwargs):
        self.id = id
        self.nome = nome
        self.vol = vol
        self.n = n
        self.e = e
        self.a = a
        self.rho = rho
        self.k = k
        self.cp = cp
        self.area = area
        self.kwargs = kwargs

class Tampa:
    def __init__(self, id, nome, vol, n, e, a, rho, k, cp, area, **kwargs):
        self.id = id
        self.nome = nome
        self.vol = vol
        self.n = n
        self.e = e
        self.a = a
        self.rho = rho
        self.k = k
        self.cp = cp
        self.area = area
        self.kwargs = kwargs

class Estrutura:
    def __init__(self, id, nome, vol, n, e, a, rho, k, cp, area, **kwargs):
        self.id = id
        self.nome = nome
        self.vol = vol
        self.n = n
        self.e = e
        self.a = a
        self.rho = rho
        self.k = k
        self.cp = cp
        self.area = area
        self.kwargs = kwargs

class PCB:
    def __init__(self, id, nome,vol, n, e, a, rho, k, cp,area, **kwargs):
        self.id = id
        self.nome = nome
        self.vol = vol
        self.n = n
        self.e = e
        self.a = a
        self.rho = rho
        self.k = k
        self.cp = cp
        self.area = area
        self.kwargs = kwargs

class Bateria:
    pass


class Parafusos:
    def __init__(self, id, nome, vol, rho, k, cp, area, **kwargs):
        self.id = id
        self.nome = nome
        self.vol = vol
        self.rho = rho
        self.k = k
        self.cp = cp
        self.e = 0
        self.a = 0
        self.area = area
        self.kwargs = kwargs

# Criar os objetos e a lista de nós

# Paineis solares
"""
Material - Silicio
rho = 2330.0 kg/m^3
k = 148,9 W/m.K
cp = 712,8 J/kg.K
a=0.578, 
e=0.557
"""
substrato1 = Substrato('1', 'substrato 1', vol=0.000002696, area=0.002696, a=0.578, e=0.557, rho=2330., k=148.9, cp=712.8, area_rad=0.002696)
substrato2 = Substrato('2', 'substrato 2', vol=0.000002696, area=0.002696, a=0.578, e=0.557, rho=2330., k=148.9, cp=712.8, area_rad=0.002696)
substrato3 = Substrato('3', 'substrato 3', vol=0.000002696, area=0.002696, a=0.578, e=0.557, rho=2330., k=148.9, cp=712.8, area_rad=0.002696)
substrato4 = Substrato('4', 'substrato 4', vol=0.000002696, area=0.002696, a=0.578, e=0.557, rho=2330., k=148.9, cp=712.8, area_rad=0.002696)
substrato5 = Substrato('5', 'substrato 5', vol=0.000002696, area=0.002696, a=0.578, e=0.557, rho=2330., k=148.9, cp=712.8, area_rad=0.002696)
substrato6 = Substrato('6', 'substrato 6', vol=0.000002696, area=0.002696, a=0.578, e=0.557, rho=2330., k=148.9, cp=712.8, area_rad=0.002696)
substrato7 = Substrato('7', 'substrato 7', vol=0.000002696, area=0.002696, a=0.578, e=0.557, rho=2330., k=148.9, cp=712.8, area_rad=0.002696)
substrato8 = Substrato('8', 'substrato 8', vol=0.000002696, area=0.002696, a=0.578, e=0.557, rho=2330., k=148.9, cp=712.8, area_rad=0.002696)

# Base dos substratos
"""
Material - FR-4.0
rho = 1850 kg/m^3
k_in = 0.29 W/m.K
k_tr = 0.81 W/m.K
cp = 1100 J/kg.K

cobertura de Optical solar reflector, indium-tin-oxide (ITO) coated
a = 0.07
e = 0.76
"""
basepainel1 = BasePainel('9', 'Base painel 1', vol=0.0000168, n='i', e=0.76, a=0.7, rho=1850.0, k=0.81,
                         cp=1100.0, area=0.0084, area_rad = 0.003008)
basepainel2 = BasePainel('10', 'Base painel 2', vol=0.0000168, n='j', e=0.76, a=0.7, rho=1850.0, k=0.81,
                         cp=1100.0, area=0.0084,area_rad = 0.003008)
basepainel3 = BasePainel('11', 'Base painel 3', vol=0.0000168, n='i', e=0.76, a=0.7, rho=1850.0, k=0.81,
                         cp=1100.0,  area=0.0084, area_rad = 0.003008)
basepainel4 = BasePainel('12', 'Base painel 4', vol=0.0000168, n='j', e=0.76, a=0.7, rho=1850.0, k=0.81,
                         cp=1100.0,  area=0.0084, area_rad = 0.003008)

# Base painel de cobre
"""
Material - Alumínio 7075
rho = 2770.0 kg/m^3
k = 121.2 W/m.K
cp = 961.2 J/kg.K
e = 0.3
a = 0.13
"""
basecobre1 = BaseCobre('13', 'base cobre 1', vol=0.0000084, n='i', e=0.3, a=0.13, rho=8580.0, k=122.9,
                       cp=378.0, area=0.0084)
basecobre2 = BaseCobre('14', 'base cobre 2', vol=0.0000084, n='j', e=0.3, a=0.13, rho=8580.0, k=122.9,
                       cp=378.0, area=0.0084)
basecobre3 = BaseCobre('15', 'base cobre 3', vol=0.0000084, n='i', e=0.3, a=0.13, rho=8580.0, k=122.9,
                       cp=378.0, area=0.0084)
basecobre4 = BaseCobre('16', 'base cobre 4', vol=0.0000084, n='j', e=0.3, a=0.13, rho=8580.0, k=122.9,
                       cp=378.0, area=0.0084)

# Tampa superior e inferior
"""
Material - Alumínio 7075
rho = 2770.0 kg/m^3
k = 121.2 W/m.K
cp = 961.2 J/kg.K
e = 0.3
a = 0.13
"""
tampa1 = Tampa('17', 'Tampa 1', vol=0.00002914719, n='k', e=0.3, a=0.13, rho=2270.0, k=121.2, cp=961.2, area=0.00971573, area_rad=0.00971573)
tampa2 = Tampa('18', 'Tampa 2', vol=0.00002914719, n='k', e=0.3, a=0.13, rho=2270.0, k=121.2, cp=961.2, area=0.00971573, area_rad=0.00971573)

# PCB's
"""
Material FR-4.0
rho = 1850 kg/m^3
k_in = 0.29 W/m.K
k_tr = 0.81 W/m.K
cp = 1100 J/kg.K

cobertura de Optical solar reflector, indium-tin-oxide (ITO) coated
a = 0.07
e = 0.76
"""
pcb1 = PCB('19', 'PCB 1', vol=0.00001911519, n='k', e=0.76, a=0.07, rho=1850.0, k=0.81, cp=1100.0, area=0.00637173)
pcb2 = PCB('20', 'PCB 2', vol=0.00001911519, n='k', e=0.76, a=0.07, rho=1850.0, k=0.81, cp=1100.0, area=0.00637173)
pcb3 = PCB('21', 'PCB 3', vol=0.00001911519, n='k', e=0.76, a=0.07, rho=1850.0, k=0.81, cp=1100.0, area=0.00637173)
pcb4 = PCB('22', 'PCB 4', vol=0.00001911519, n='k', e=0.76, a=0.07, rho=1850.0, k=0.81, cp=1100.0, area=0.00637173)

# Estruturas Verticais
"""
Material - Alumínio 7075
rho = 2770.0 kg/m^3
k = 121.2 W/m.K
cp = 961.2 J/kg.K
e = 0.3
a = 0.13
"""
estrutura_vertical1 = Estrutura('23', 'Estrutura Vertical 1', vol=0.000028, n='i', e=0.3, a=0.5,
                                rho=2770.0, k=121.2, cp=961.2, area=0.0008, area_rad=0.0008)
estrutura_vertical2 = Estrutura('24', 'Estrutura Vertical 2', vol=0.000028, n='j', e=0.3, a=0.5,
                                rho=2770.0, k=121.2, cp=961.2, area=0.0008, area_rad=0.0008)
estrutura_vertical3 = Estrutura('25', 'Estrutura Vertical 3', vol=0.000028, n='i', e=0.3, a=0.5,
                                rho=2770.0, k=121.2, cp=961.2, area=0.0008, area_rad=0.0008)
estrutura_vertical4 = Estrutura('26', 'Estrutura Vertical 4', vol=0.000028, n='j', e=0.3, a=0.5,
                                rho=2770.0, k=121.2, cp=961.2, area=0.0008, area_rad=0.0008)
estrutura_vertical5 = Estrutura('27', 'Estrutura Vertical 5', vol=0.000028, n='i', e=0.3, a=0.5,
                                rho=2770.0, k=121.2, cp=961.2, area=0.0008, area_rad=0.0008)
estrutura_vertical6 = Estrutura('28', 'Estrutura Vertical 6', vol=0.000028, n='j', e=0.3, a=0.5,
                                rho=2770.0, k=121.2, cp=961.2, area=0.0008, area_rad=0.0008)
estrutura_vertical7 = Estrutura('29', 'Estrutura Vertical 7', vol=0.000028, n='i', e=0.3, a=0.5,
                                rho=2770.0, k=121.2, cp=961.2, area=0.0008, area_rad=0.0008)
estrutura_vertical8 = Estrutura('30', 'Estrutura Vertical 8', vol=0.000028, n='j', e=0.3, a=0.5,
                                rho=2770.0, k=121.2, cp=961.2, area=0.0008, area_rad=0.0008)

# Estruturas Horizontais
"""
Material - Alumínio 7075
rho = 2770.0 kg/m^3
k = 121.2 W/m.K
cp = 961.2 J/kg.K
e = 0.3
a = 0.13
"""
estrutura_horizontal1 = Estrutura('31', 'Estrutura Horizontal 1', vol=0.000002352, n='i', e=0.3, a=0.13,
                                  rho=2770.0, k=121.2, cp=961.2, area=0.0008)
estrutura_horizontal2 = Estrutura('32', 'Estrutura Horizontal 2', vol=0.000002352, n='i', e=0.3, a=0.13,
                                  rho=2770.0, k=121.2, cp=961.2, area=0.0008)
estrutura_horizontal3 = Estrutura('33', 'Estrutura Horizontal 3', vol=0.000002352, n='j', e=0.3, a=0.13,
                                  rho=2770.0, k=121.2, cp=961.2, area=0.0008)
estrutura_horizontal4 = Estrutura('34', 'Estrutura Horizontal 4', vol=0.000002352, n='j', e=0.3, a=0.13,
                                  rho=2770.0, k=121.2, cp=961.2, area=0.0008)
estrutura_horizontal5 = Estrutura('35', 'Estrutura Horizontal 5', vol=0.000002352, n='i', e=0.3, a=0.13,
                                  rho=2770.0, k=121.2, cp=961.2, area=0.0008)
estrutura_horizontal6 = Estrutura('36', 'Estrutura Horizontal 6', vol=0.000002352, n='i', e=0.3, a=0.13,
                                  rho=2770.0, k=121.2, cp=961.2, area=0.0008)
estrutura_horizontal7 = Estrutura('37', 'Estrutura Horizontal 7', vol=0.000002352, n='j', e=0.3, a=0.13,
                                  rho=2770.0, k=121.2, cp=961.2, area=0.0008)
estrutura_horizontal8 = Estrutura('38', 'Estrutura Horizontal 8', vol=0.000002352, n='j', e=0.3, a=0.13,
                                  rho=2770.0, k=121.2, cp=961.2, area=0.0008)

estrutura_horizontal9 = Estrutura('39', 'Estrutura Horizontal 9', vol=0.000002352, n='k', e=0.3, a=0.13,
                                  rho=2770.0, k=121.2, cp=961.2, area=0.0008)
estrutura_horizontal10 = Estrutura('40', 'Estrutura Horizontal 10', vol=0.000002352, n='k', e=0.3, a=0.13,
                                   rho=2770.0, k=121.2, cp=961.2, area=0.0008)
estrutura_horizontal11 = Estrutura('41', 'Estrutura Horizontal 11', vol=0.000002352, n='k', e=0.3, a=0.13,
                                   rho=2770.0, k=121.2, cp=961.2, area=0.0008)
estrutura_horizontal12 = Estrutura('42', 'Estrutura Horizontal 12', vol=0.000002352, n='k', e=0.3, a=0.13,
                                   rho=2770.0, k=121.2, cp=961.2, area=0.0008)
estrutura_horizontal13 = Estrutura('43', 'Estrutura Horizontal 13', vol=0.000002352, n='k', e=0.3, a=0.13,
                                   rho=2770.0, k=121.2, cp=961.2, area=0.0008)
estrutura_horizontal14 = Estrutura('44', 'Estrutura Horizontal 14', vol=0.000002352, n='k', e=0.3, a=0.13,
                                   rho=2770.0, k=121.2, cp=961.2, area=0.0008)
estrutura_horizontal15 = Estrutura('45', 'Estrutura Horizontal 15', vol=0.000002352, n='k', e=0.3, a=0.13,
                                   rho=2770.0, k=121.2, cp=961.2, area=0.0008)
estrutura_horizontal16 = Estrutura('46', 'Estrutura Horizontal 16', vol=0.000002352, n='k', e=0.3, a=0.13,
                                   rho=2770.0, k=121.2, cp=961.2, area=0.0008)

# Parafusos
"""
Material - Alumínio 7075
rho = 2770.0 kg/m^3
k = 121.2 W/m.K
cp = 961.2 J/kg.K
e = 0.3
a = 0.13
"""
parafuso1 = Parafusos('47', 'Parafuso 1', vol=0.0000020412, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso2 = Parafusos('48', 'Parafuso 2', vol=0.0000020412, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso3 = Parafusos('49', 'Parafuso 3', vol=0.0000020412, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso4 = Parafusos('50', 'Parafuso 4', vol=0.0000020412, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso5 = Parafusos('51', 'Parafuso 5', vol=0.0000020412, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso6 = Parafusos('52', 'Parafuso 6', vol=0.0000020412, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso7 = Parafusos('53', 'Parafuso 7', vol=0.0000020412, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso8 = Parafusos('54', 'Parafuso 8', vol=0.0000020412, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso9 = Parafusos('55', 'Parafuso 9', vol=0.0000020412, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso10 = Parafusos('56', 'Parafuso 10', vol=0.0000002856, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso11 = Parafusos('57', 'Parafuso 11', vol=0.0000002856, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso12 = Parafusos('58', 'Parafuso 12', vol=0.0000002856, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso13 = Parafusos('59', 'Parafuso 13', vol=0.0000002856, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso14 = Parafusos('60', 'Parafuso 14', vol=0.0000002856, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso15 = Parafusos('61', 'Parafuso 15', vol=0.0000002856, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso16 = Parafusos('62', 'Parafuso 16', vol=0.0000002856, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso17 = Parafusos('63', 'Parafuso 17', vol=0.0000002856, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso18 = Parafusos('64', 'Parafuso 18', vol=0.0000002856, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso19 = Parafusos('65', 'Parafuso 19', vol=0.0000002856, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)
parafuso20 = Parafusos('66', 'Parafuso 20', vol=0.0000002856, rho=2770.0, k=121.2, cp=961.2, area=0.00034644)

nos = [substrato1, substrato2, substrato3, substrato4, substrato5, substrato6, substrato7, substrato8,
       basepainel1, basepainel2, basepainel3, basepainel4, basecobre1, basecobre2, basecobre3, basecobre4,
       tampa1, tampa2,
       pcb1, pcb2, pcb3, pcb4,
       estrutura_vertical1, estrutura_vertical2, estrutura_vertical3, estrutura_vertical4, estrutura_vertical5,
       estrutura_vertical6, estrutura_vertical7, estrutura_vertical8,
       estrutura_horizontal1, estrutura_horizontal2, estrutura_horizontal3, estrutura_horizontal4,
       estrutura_horizontal5, estrutura_horizontal6,
       estrutura_horizontal7, estrutura_horizontal8, estrutura_horizontal9, estrutura_horizontal10,
       estrutura_horizontal11, estrutura_horizontal12,
       estrutura_horizontal13, estrutura_horizontal14, estrutura_horizontal15, estrutura_horizontal16,
       parafuso1, parafuso2, parafuso3, parafuso4, parafuso5, parafuso6, parafuso7, parafuso8, parafuso9, parafuso10,
       parafuso11, parafuso12, parafuso13, parafuso14, parafuso15, parafuso16, parafuso17, parafuso18, parafuso19,
       parafuso20]

# Conexões de condução entre os nós

conduc = {substrato1.id: [basepainel1.id],
          substrato2.id: [basepainel1.id],
          basepainel1.id: [substrato1.id, substrato2.id, basecobre1.id],
          basecobre1.id: [basepainel1.id, estrutura_horizontal1.id, estrutura_horizontal2.id],
          substrato3.id: [basepainel2.id],
          substrato4.id: [basepainel2.id],
          basepainel2.id: [substrato3.id, substrato4.id, basecobre2.id],
          basecobre2.id: [basepainel2.id, estrutura_horizontal3.id, estrutura_horizontal4.id],
          substrato5.id: [basepainel3.id],
          substrato6.id: [basepainel3.id],
          basepainel3.id: [substrato5.id, substrato6.id, basecobre3.id],
          basecobre3.id: [basepainel3.id, estrutura_horizontal5.id, estrutura_horizontal6.id],
          substrato7.id: [basepainel4.id],
          substrato8.id: [basepainel4.id],
          basepainel4.id: [substrato7.id, substrato8.id, basecobre4.id],
          basecobre4.id: [basepainel4.id, estrutura_horizontal7.id, estrutura_horizontal8.id],

          estrutura_horizontal1.id: [basecobre1.id, estrutura_horizontal9.id, estrutura_vertical1.id,
                                     estrutura_vertical8.id],
          estrutura_horizontal2.id: [basecobre1.id, estrutura_horizontal10.id, estrutura_vertical1.id,
                                     estrutura_vertical8.id],
          estrutura_horizontal3.id: [basecobre2.id, estrutura_horizontal11.id, estrutura_vertical3.id,
                                     estrutura_vertical2.id],
          estrutura_horizontal4.id: [basecobre2.id, estrutura_horizontal12.id, estrutura_vertical3.id,
                                     estrutura_vertical2.id],
          estrutura_horizontal5.id: [basecobre3.id, estrutura_horizontal13.id, estrutura_vertical5.id,
                                     estrutura_vertical4.id],
          estrutura_horizontal6.id: [basecobre3.id, estrutura_horizontal14.id, estrutura_vertical5.id,
                                     estrutura_vertical4.id],
          estrutura_horizontal7.id: [basecobre4.id, estrutura_horizontal15.id, estrutura_vertical6.id,
                                     estrutura_vertical7.id],
          estrutura_horizontal8.id: [basecobre4.id, estrutura_horizontal16.id, estrutura_vertical6.id,
                                     estrutura_vertical7.id],

          tampa2.id: [estrutura_horizontal9.id, estrutura_horizontal11.id, estrutura_horizontal13.id,
                      estrutura_horizontal15.id, parafuso1.id, parafuso2.id, parafuso3.id, parafuso4.id],
          tampa1.id: [estrutura_horizontal10.id, estrutura_horizontal12.id, estrutura_horizontal14.id,
                      estrutura_horizontal16.id, parafuso5.id, parafuso6.id, parafuso7.id, parafuso8.id],

          parafuso1.id: [tampa2.id, pcb1.id],
          parafuso2.id: [tampa2.id, pcb1.id],
          parafuso3.id: [tampa2.id, pcb1.id],
          parafuso4.id: [tampa2.id, pcb1.id],
          parafuso5.id: [tampa1.id, pcb4.id],
          parafuso6.id: [tampa1.id, pcb4.id],
          parafuso7.id: [tampa1.id, pcb4.id],
          parafuso8.id: [tampa1.id, pcb4.id],
          parafuso9.id: [pcb1.id, pcb2.id],
          parafuso10.id: [pcb1.id, pcb2.id],
          parafuso11.id: [pcb1.id, pcb2.id],
          parafuso12.id: [pcb1.id, pcb2.id],
          parafuso13.id: [pcb2.id, pcb3.id],
          parafuso14.id: [pcb2.id, pcb3.id],
          parafuso15.id: [pcb2.id, pcb3.id],
          parafuso16.id: [pcb2.id, pcb3.id],
          parafuso17.id: [pcb3.id, pcb4.id],
          parafuso18.id: [pcb3.id, pcb4.id],
          parafuso19.id: [pcb3.id, pcb4.id],
          parafuso20.id: [pcb3.id, pcb4.id],

          pcb1.id: [parafuso1.id, parafuso2.id, parafuso3.id, parafuso4.id, parafuso9.id, parafuso10.id, parafuso11.id,
                    parafuso12.id],
          pcb2.id: [parafuso9.id, parafuso10.id, parafuso11.id, parafuso12.id, parafuso13.id, parafuso14.id,
                    parafuso15.id, parafuso16.id],
          pcb3.id: [parafuso13.id, parafuso14.id, parafuso15.id, parafuso16.id, parafuso17.id, parafuso18.id,
                    parafuso19.id, parafuso20.id],
          pcb4.id: [parafuso17.id, parafuso18.id, parafuso19.id, parafuso20.id, parafuso5.id, parafuso6.id,
                    parafuso7.id, parafuso8.id],

          estrutura_vertical1.id: [estrutura_horizontal1.id, estrutura_horizontal2.id, estrutura_vertical2.id],
          estrutura_vertical2.id: [estrutura_horizontal3.id, estrutura_horizontal4.id, estrutura_vertical1.id],
          estrutura_vertical3.id: [estrutura_horizontal3.id, estrutura_horizontal4.id, estrutura_vertical4.id],
          estrutura_vertical4.id: [estrutura_horizontal5.id, estrutura_horizontal6.id, estrutura_vertical3.id],
          estrutura_vertical5.id: [estrutura_horizontal5.id, estrutura_horizontal6.id, estrutura_vertical6.id],
          estrutura_vertical6.id: [estrutura_horizontal7.id, estrutura_horizontal8.id, estrutura_vertical5.id],
          estrutura_vertical7.id: [estrutura_horizontal7.id, estrutura_horizontal8.id, estrutura_vertical8.id],
          estrutura_vertical8.id: [estrutura_horizontal1.id, estrutura_horizontal2.id, estrutura_vertical7.id],

          estrutura_horizontal9.id: [tampa2.id, estrutura_horizontal1.id, estrutura_vertical1.id, estrutura_vertical8.id],
          estrutura_horizontal10.id: [tampa1.id, estrutura_horizontal2.id, estrutura_vertical1.id, estrutura_vertical8.id],
          estrutura_horizontal11.id: [tampa2.id, estrutura_horizontal3.id, estrutura_vertical3.id, estrutura_vertical2.id],
          estrutura_horizontal12.id: [tampa1.id, estrutura_horizontal4.id, estrutura_vertical3.id, estrutura_vertical2.id],
          estrutura_horizontal13.id: [tampa2.id, estrutura_horizontal5.id, estrutura_vertical4.id, estrutura_vertical5.id],
          estrutura_horizontal14.id: [tampa1.id, estrutura_horizontal6.id, estrutura_vertical4.id, estrutura_vertical5.id],
          estrutura_horizontal15.id: [tampa2.id, estrutura_horizontal7.id, estrutura_vertical6.id, estrutura_vertical7.id],
          estrutura_horizontal16.id: [tampa1.id, estrutura_horizontal8.id, estrutura_vertical6.id, estrutura_vertical7.id]
          }


def ordena_id(lista):
    a = list(lista.keys())
    b = []
    for valor in a:
        b.append(int(valor))
    b.sort()
    lista_ids = []
    for valor in b:
        lista_ids.append(str(valor))
    return lista_ids


lista_ids = ordena_id(conduc)

# Define a matriz inicial preenchida por zeros

K_m = np.zeros((len(conduc), len(conduc)))

# Lógica para preencher a matriz com 1 onde existe condução entre nós

for chave in conduc.keys():
    A = (conduc[chave])
    for valor in A:
        K_m[int(chave) - 1, int(valor) - 1] = 1


def nome(nos, id):
    for no in nos:
        if no.id == id:
            A = no.nome
    return A


# Disntância entre nós
ids = list(conduc.keys())
valores = list(conduc.values())
'''L = {}
A = {}
j = 0
for valor in valores:
    for i in range(len(valor)):
        L[" ".join([ids[j], valor[i]])] = input(f'Comprimento de {nome(nos,ids[j])} a {nome(nos, valor[i])} [m]: ')
        #A[" ".join([ids[j], valor[i]])] = input(f'Area entre {nome(nos,ids[j])} e {nome(nos, valor[i])} [m^2]: ')
    j += 1
print(L)'''

L = {'1 9': '0.0005', '2 9': '0.0005', '9 1': '0.001', '9 2': '0.001', '9 13': '0.001', '13 9': '0.0005',
     '13 31': '0.0005', '13 32': '0.0005', '3 10': '0.0005', '4 10': '0.0005', '10 3': '0.001', '10 4': '0.001',
     '10 14': '0.001', '14 10': '0.0005', '14 33': '0.0005', '14 34': '0.0005', '5 11': '0.0005', '6 11': '0.0005',
     '11 5': '0.001', '11 6': '0.001', '11 15': '0.001', '15 11': '0.0005', '15 35': '0.0005', '15 36': '0.0005',
     '7 12': '0.0005', '8 12': '0.0005', '12 7': '0.001', '12 8': '0.001', '12 16': '0.001', '16 12': '0.0005',
     '16 37': '0.0005', '16 38': '0.0005', '31 13': '0.001', '31 39': '0.004', '31 23': '0.042', '31 30': '0.042',
     '32 13': '0.001', '32 40': '0.004', '32 23': '0.042', '32 30': '0.042', '33 14': '0.001', '33 41': '0.004',
     '33 25': '0.042', '33 24': '0.042', '34 14': '0.001', '34 42': '0.004', '34 25': '0.042', '34 24': '0.042',
     '35 15': '0.001', '35 43': '0.004', '35 27': '0.042', '35 26': '0.042', '36 15': '0.001', '36 44': '0.004',
     '36 27': '0.042', '36 26': '0.042', '37 16': '0.001', '37 45': '0.004', '37 28': '0.042', '37 29': '0.042',
     '38 16': '0.001', '38 46': '0.004', '38 28': '0.042', '38 29': '0.042', '18 39': '0.042', '18 41': '0.042',
     '18 43': '0.042', '18 45': '0.042', '18 47': '0.04242905', '18 48': '0.04242905', '18 49': '0.04242905',
     '18 50': '0.04242905', '17 40': '0.042', '17 42': '0.042', '17 44': '0.042', '17 46': '0.042',
     '17 51': '0.04242905', '17 52': '0.04242905', '17 53': '0.04242905', '17 54': '0.04242905', '47 18': '0.007',
     '47 19': '0.007', '48 18': '0.007', '48 19': '0.007', '49 18': '0.007', '49 19': '0.007', '50 18': '0.007',
     '50 19': '0.007', '51 17': '0.007', '51 22': '0.007', '52 17': '0.007', '52 22': '0.007', '53 17': '0.007',
     '53 22': '0.007', '54 17': '0.007', '54 22': '0.007', '55 19': '0.01', '55 20': '0.01', '56 19': '0.01',
     '56 20': '0.01', '57 19': '0.01', '57 20': '0.01', '58 19': '0.01', '58 20': '0.01', '59 20': '0.01',
     '59 21': '0.01', '60 20': '0.01', '60 21': '0.01', '61 20': '0.01', '61 21': '0.01', '62 20': '0.01',
     '62 21': '0.01', '63 21': '0.01', '63 22': '0.01', '64 21': '0.01', '64 22': '0.01', '65 21': '0.01',
     '65 22': '0.01', '66 21': '0.01', '66 22': '0.01', '19 47': '0.04242905', '19 48': '0.04242905',
     '19 49': '0.04242905', '19 50': '0.04242905', '19 55': '0.04242905', '19 56': '0.04242905', '19 57': '0.04242905',
     '19 58': '0.04242905', '20 55': '0.04242905', '20 56': '0.04242905', '20 57': '0.04242905', '20 58': '0.04242905',
     '20 59': '0.04242905', '20 60': '0.04242905', '20 61': '0.04242905', '20 62': '0.04242905', '21 59': '0.04242905',
     '21 60': '0.04242905', '21 61': '0.04242905', '21 62': '0.04242905', '21 63': '0.04242905', '21 64': '0.04242905',
     '21 65': '0.04242905', '21 66': '0.04242905', '22 63': '0.04242905', '22 64': '0.04242905', '22 65': '0.04242905',
     '22 66': '0.04242905', '22 51': '0.04242905', '22 52': '0.04242905', '22 53': '0.04242905', '22 54': '0.04242905',
     '23 31': '0.05', '23 32': '0.05', '23 24': '0.004', '24 33': '0.05', '24 34': '0.05', '24 23': '0.004',
     '25 33': '0.05', '25 34': '0.05', '25 26': '0.004', '26 35': '0.05', '26 36': '0.05', '26 25': '0.004',
     '27 35': '0.05', '27 36': '0.05', '27 28': '0.004', '28 37': '0.05', '28 38': '0.05', '28 27': '0.004',
     '29 37': '0.05', '29 38': '0.05', '29 30': '0.004', '30 31': '0.05', '30 32': '0.05', '30 29': '0.004',
     '39 18': '0.001', '39 31': '0.004', '39 23': '0.042', '39 30': '0.042', '40 17': '0.001', '40 32': '0.004',
     '40 23': '0.042', '40 30': '0.042', '41 18': '0.001', '41 33': '0.004', '41 25': '0.042', '41 24': '0.042',
     '42 17': '0.001', '42 34': '0.004', '42 25': '0.042', '42 24': '0.042', '43 18': '0.001', '43 35': '0.004',
     '43 26': '0.042', '43 27': '0.042', '44 17': '0.001', '44 36': '0.004', '44 26': '0.042', '44 27': '0.042',
     '45 18': '0.001', '45 37': '0.004', '45 28': '0.042', '45 29': '0.042', '46 17': '0.001', '46 38': '0.004',
     '46 28': '0.042', '46 29': '0.042'}

'''A = {}
j = 0
for valor in valores:
    for i in range(len(valor)):
        #L[" ".join([ids[j], valor[i]])] = input(f'Comprimento de {nome(nos,ids[j])} a {nome(nos, valor[i])} [m]: ')
        A[" ".join([ids[j], valor[i]])] = input(f'Area entre {nome(nos,ids[j])} e {nome(nos, valor[i])} [m^2]: ')
    j += 1
print(A)'''


A = {'1 9': '0.002696', '2 9': '0.002696', '9 1': '0.002696', '9 2': '0.002696', '9 13': '0.0084', '13 9': '0.0084',
     '13 31': '0.000672', '13 32': '0.000672', '3 10': '0.002696', '4 10': '0.002696', '10 3': '0.002696',
     '10 4': '0.002696', '10 14': '0.0084', '14 10': '0.0084', '14 33': '0.000672', '14 34': '0.000672',
     '5 11': '0.002696', '6 11': '0.002696', '11 5': '0.002696', '11 6': '0.002696', '11 15': '0.0084',
     '15 11': '0.0084', '15 35': '0.000672', '15 36': '0.000672', '7 12': '0.002696', '8 12': '0.002696',
     '12 7': '0.002696', '12 8': '0.002696', '12 16': '0.0084', '16 12': '0.0084', '16 37': '0.000672',
     '16 38': '0.000672', '31 13': '0.000672', '31 39': '0.000168', '31 23': '0.00004', '31 30': '0.00004',
     '32 13': '0.000672', '32 40': '0.000168', '32 23': '0.00004', '32 30': '0.00004', '33 14': '0.000672',
     '33 41': '0.000168', '33 25': '0.00004', '33 24': '0.00004', '34 14': '0.000672', '34 42': '0.000168',
     '34 25': '0.00004', '34 24': '0.00004', '35 15': '0.000672', '35 43': '0.000168', '35 27': '0.00004',
     '35 26': '0.00004', '36 15': '0.000672', '36 44': '0.000168', '36 27': '0.00004', '36 26': '0.00004',
     '37 16': '0.000672', '37 45': '0.000168', '37 28': '0.00004', '37 29': '0.00004', '38 16': '0.000672',
     '38 46': '0.000168', '38 28': '0.00004', '38 29': '0.00004', '18 39': '0.000672', '18 41': '0.000672',
     '18 43': '0.000672', '18 45': '0.000672', '18 47': '0.00001458', '18 48': '0.00001458', '18 49': '0.00001458',
     '18 50': '0.00001458', '17 40': '0.000672', '17 42': '0.000672', '17 44': '0.000672', '17 46': '0.000672',
     '17 51': '0.00001458', '17 52': '0.00001458', '17 53': '0.00001458', '17 54': '0.00001458', '47 18': '0.00001458',
     '47 19': '0.00001458', '48 18': '0.00001458', '48 19': '0.00001458', '49 18': '0.00001458', '49 19': '0.00001458',
     '50 18': '0.00001458', '50 19': '0.00001458', '51 17': '0.00001458', '51 22': '0.00001458', '52 17': '0.00001458',
     '52 22': '0.00001458', '53 17': '0.00001458', '53 22': '0.00001458', '54 17': '0.00001458', '54 22': '0.00001458',
     '55 19': '0.00001458', '55 20': '0.00001458', '56 19': '0.00001458', '56 20': '0.00001458', '57 19': '0.00001458',
     '57 20': '0.00001458', '58 19': '0.00001458', '58 20': '0.00001458', '59 20': '0.00001458', '59 21': '0.00001458',
     '60 20': '0.00001458', '60 21': '0.00001458', '61 20': '0.00001458', '61 21': '0.00001458', '62 20': '0.00001458',
     '62 21': '0.00001458', '63 21': '0.00001458', '63 22': '0.00001458', '64 21': '0.00001458', '64 22': '0.00001458',
     '65 21': '0.00001458', '65 22': '0.00001458', '66 21': '0.00001458', '66 22': '0.00001458', '19 47': '0.00001458',
     '19 48': '0.00001458', '19 49': '0.00001458', '19 50': '0.00001458', '19 55': '0.00001458', '19 56': '0.00001458',
     '19 57': '0.00001458', '19 58': '0.00001458', '20 55': '0.00001458', '20 56': '0.00001458', '20 57': '0.00001458',
     '20 58': '0.00001458', '20 59': '0.00001458', '20 60': '0.00001458', '20 61': '0.00001458', '20 62': '0.00001458',
     '21 59': '0.00001458', '21 60': '0.00001458', '21 61': '0.00001458', '21 62': '0.00001458', '21 63': '0.00001458',
     '21 64': '0.00001458', '21 65': '0.00001458', '21 66': '0.00001458', '22 63': '0.00001458', '22 64': '0.00001458',
     '22 65': '0.00001458', '22 66': '0.00001458', '22 51': '0.00001458', '22 52': '0.00001458', '22 53': '0.00001458',
     '22 54': '0.00001458', '23 31': '0.00004', '23 32': '0.00004', '23 24': '0.000224', '24 33': '0.00004',
     '24 34': '0.00004', '24 23': '0.000224', '25 33': '0.00004', '25 34': '0.00004', '25 26': '0.000224',
     '26 35': '0.00004', '26 36': '0.00004', '26 25': '0.000224', '27 35': '0.00004', '27 36': '0.00004',
     '27 28': '0.000224', '28 37': '0.00004', '28 38': '0.00004', '28 27': '0.000224', '29 37': '0.00004',
     '29 38': '0.00004', '29 30': '0.000224', '30 31': '0.00004', '30 32': '0.00004', '30 29': '0.000224',
     '39 18': '0.000672', '39 31': '0.000168', '39 23': '0.000224', '39 30': '0.000224', '40 17': '0.000672',
     '40 32': '0.000168', '40 23': '0.000224', '40 30': '0.000224', '41 18': '0.000672', '41 33': '0.000168',
     '41 25': '0.000224', '41 24': '0.000224', '42 17': '0.000672', '42 34': '0.000168', '42 25': '0.000224',
     '42 24': '0.000224', '43 18': '0.000672', '43 35': '0.000168', '43 26': '0.000224', '43 27': '0.000224',
     '44 17': '0.000672', '44 36': '0.000168', '44 26': '0.000224', '44 27': '0.000224', '45 18': '0.000672',
     '45 37': '0.000168', '45 28': '0.000224', '45 29': '0.000224', '46 17': '0.000672', '46 38': '0.000168',
     '46 28': '0.000224', '46 29': '0.000224'}



lista_nomes = list(A.keys())

chaves = list(A.values())
comprimento = list(L.values())

K = 0
for nome in lista_nomes:
    An = nome.split()
    i, j = An
    i = int(i)
    j = int(j)
    b = chaves[K]
    k = (float(b) * nos[i - 1].k) / float(comprimento[K])
    K_m[i - 1, j - 1] = k
    K += 1

import pandas as pd
df = pd.DataFrame(K_m)
df.to_csv("k_m.csv", sep=',')


# Definir as Cavidades de Radiação

class Cavidade:

    def __init__(self, Lx, Ly, Lz):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

class CavidadeFace:

    def __init__(self, X, Y, n, e, comp):
        self.X = X
        self.Y = Y
        self.n = n
        self.e = e
        self.comp = comp
        self.Area = X*Y

# Definição das cavidades

cavidade1 = Cavidade(0.08, 0.08, 0.014)
cavidade2 = Cavidade(0.08, 0.08, 0.02)
cavidade3 = Cavidade(0.08, 0.08, 0.02)
cavidade4 = Cavidade(0.08, 0.08, 0.02)
cavidade5 = Cavidade(0.08, 0.08, 0.014)

# Definição dos lados das cavidades

# Cavidade 1

face1 = CavidadeFace(X=0.08, Y=0.014, n='i', e=basecobre1.e, comp=basecobre1.id)
face2 = CavidadeFace(X=0.08, Y=0.014, n='j', e=basecobre2.e, comp=basecobre2.id)
face3 = CavidadeFace(X=0.08, Y=0.014, n='i', e=basecobre3.e, comp=basecobre3.id)
face4 = CavidadeFace(X=0.08, Y=0.014, n='j', e=basecobre4.e, comp=basecobre4.id)
face5 = CavidadeFace(X=0.08, Y=0.08, n='k', e=tampa2.e, comp=tampa2.id)
face6 = CavidadeFace(X=0.08, Y=0.08, n='k', e=pcb1.e, comp=pcb1.id)

C1 = [face1, face2, face3, face4, face5, face6]
fb1 = ff_interno(C1, cavidade1)# fator_gebhart2(C1, cavidade1)

# Cavidade 2

face7 = CavidadeFace(X=0.08, Y=0.02, n='i', e=basecobre1.e, comp=basecobre1.id)
face8 = CavidadeFace(X=0.08, Y=0.02, n='j', e=basecobre2.e, comp=basecobre2.id)
face9 = CavidadeFace(X=0.08, Y=0.02, n='i', e=basecobre3.e, comp=basecobre3.id)
face10 = CavidadeFace(X=0.08, Y=0.02, n='j', e=basecobre4.e, comp=basecobre4.id)
face11 = CavidadeFace(X=0.08, Y=0.08, n='k', e=pcb1.e, comp=pcb1.id)
face12 = CavidadeFace(X=0.08, Y=0.08, n='k', e=pcb2.e, comp=pcb2.id)

C2 = [face7, face8, face9, face10, face11, face12]
fb2 = ff_interno(C2, cavidade2)# fator_gebhart2(C2, cavidade2)
# Cavidade 3

face13 = CavidadeFace(X=0.08, Y=0.02, n='i', e=basecobre1.e, comp=basecobre1.id)
face14 = CavidadeFace(X=0.08, Y=0.02, n='j', e=basecobre2.e, comp=basecobre2.id)
face15 = CavidadeFace(X=0.08, Y=0.02, n='i', e=basecobre3.e, comp=basecobre3.id)
face16 = CavidadeFace(X=0.08, Y=0.02, n='j', e=basecobre4.e, comp=basecobre4.id)
face17 = CavidadeFace(X=0.08, Y=0.08, n='k', e=pcb2.e, comp=pcb2.id)
face18 = CavidadeFace(X=0.08, Y=0.08, n='k', e=pcb3.e, comp=pcb3.id)

C3 = [face13, face14, face15, face16, face17, face18]
fb3 = ff_interno(C3, cavidade3) # fator_gebhart2(C3, cavidade3)
# Cavidade 4

face19 = CavidadeFace(X=0.08, Y=0.02, n='i', e=basecobre1.e, comp=basecobre1.id)
face20 = CavidadeFace(X=0.08, Y=0.02, n='j', e=basecobre2.e, comp=basecobre2.id)
face21 = CavidadeFace(X=0.08, Y=0.02, n='i', e=basecobre3.e, comp=basecobre3.id)
face22 = CavidadeFace(X=0.08, Y=0.02, n='j', e=basecobre4.e, comp=basecobre4.id)
face23 = CavidadeFace(X=0.08, Y=0.08, n='k', e=pcb3.e, comp=pcb3.id)
face24 = CavidadeFace(X=0.08, Y=0.08, n='k', e=pcb4.e, comp=pcb4.id)

C4 = [face19, face20, face21, face22, face23, face24]
fb4 = ff_interno(C4, cavidade4) #fator_gebhart2(C4, cavidade4)
# Cavidade 5

face25 = CavidadeFace(X=0.08, Y=0.014, n='i', e=basecobre1.e, comp=basecobre1.id)
face26 = CavidadeFace(X=0.08, Y=0.014, n='j', e=basecobre2.e, comp=basecobre2.id)
face27 = CavidadeFace(X=0.08, Y=0.014, n='i', e=basecobre3.e, comp=basecobre3.id)
face28 = CavidadeFace(X=0.08, Y=0.014, n='j', e=basecobre4.e, comp=basecobre4.id)
face29 = CavidadeFace(X=0.08, Y=0.08, n='k', e=pcb4.e, comp=pcb4.id)
face30 = CavidadeFace(X=0.08, Y=0.08, n='k', e=tampa1.e, comp=tampa1.id)

C5 = [face25, face26, face27, face28, face29, face30]
fb5 = ff_interno(C5, cavidade5) #fator_gebhart2(C5, cavidade5)

# preencher a matriz com radiações
lista_rad = [C1, C2, C3, C4, C5]
fg1 = [fb1, fb2, fb3, fb4, fb5]
lista_fg = []
for fb in fg1:
    for item in fb:
        for dado in item:
            lista_fg.append(dado)

rad = np.zeros((len(conduc), len(conduc)))
indices = []
for cavidade in lista_rad:
    for face1 in cavidade:
        for face2 in cavidade:
            indices.append([int(face1.comp), int(face2.comp)])
k = 0
for indice in indices:
    i, j = indice
    rad[i-1, j-1] = lista_fg[k]
    k += 1

import pandas as pd
df = pd.DataFrame(rad)
df.to_csv("rad.csv", sep=',')
# dict com os nos que receberão radiação

nos_rad = {'face 1': [substrato1.id, substrato2.id, basepainel1.id, estrutura_vertical1.id, estrutura_vertical8.id],
           'face 2': [substrato3.id, substrato4.id, basepainel2.id, estrutura_vertical2.id, estrutura_vertical3.id],
           'face 3': [substrato5.id, substrato6.id, basepainel3.id, estrutura_vertical4.id, estrutura_vertical5.id],
           'face 4': [substrato7.id, substrato8.id, basepainel4.id, estrutura_vertical6.id, estrutura_vertical7.id],
           'face 5': [tampa1.id],
           'face 6': [tampa2.id]
}


#df = pd.DataFrame(calor2, columns=lista_ids )

from plots import *

#df.to_csv('calor.csv', sep=',')

# calcular time step transiente

def deltaT(fg, K_m, nos):
    delt = []
    soma = []
    for i in range(0,len(fg)):
        for j in range(0, len(K_m)):
            a = ((fg[i,j] + K_m[i,j]) * nos[i].rho *  nos[i].vol * nos[i].cp)
            if a != 0:
                soma.append(1 / (fg[i,j] + K_m[i,j]))
        delt.append(1 / (sum(soma.copy()) * nos[i].rho *  nos[i].vol * nos[i].cp))
        soma = []
    return min(delt)

delt = deltaT(rad, K_m, nos)

# motor de calculo

def lumped(calor, fg, area_rad_ext, nos, K_m, beta):

    # Temperaturas dos nós iniciais que serão alteradas a cada passo de integração

    Temp_ini = [300.0 for i in range(0, len(nos))]
    Temp_pos = [300.0 for i in range(0, len(nos))]
    Temp_ini_ext = [0 for i in range(0, len(nos))]
    Nos_rad = list(nos_rad.values())
    for lista_no in Nos_rad:
        for no in lista_no:
            j = int(no)
            Temp_ini_ext[j-1] = Temp_ini[j-1]

    # passo de integracao termica
    Dt_termico = 0.01 #deltaT(fg, K_m, nos)

    # passo de integracao da orbita
    Dt_orbita = 1.0

    # constante de steffan boltzmann
    sig = 5.67e-8
    cores = ['#5F9EA0', '#66CDAA', '#7FFFD4', '#006400', '#556B2F', '#8FBC8F', '#2E8B57', '#3CB371', '#20B2AA', '#98FB98',
             '#00FF7F', '#7CFC00', '#00FF00', '#7FFF00', '#00FA9A', '#ADFF2F', '#32CD32', '#9ACD32', '#228B22', '#6B8E23',
             '#BDB76B','#FF6347','#FF4500','#FF0000','#FF69B4','#FF1493','#FFC0CB','#FFB6C1','#DB7093',
             '#B03060','#C71585','#D02090','#FF00FF','#EE82EE','#DDA0DD','#DA70D6','#BA55D3',
             '#9932CC','#9400D3','#8A2BE2','#A020F0','#9370DB','#D8BFD8','#FFFAFA']
    total = len(calor)

    cont = 0
    n = 0
    numero_orbitas = 15
    # calculo da temperaturas
    var_temp = []
    # tracemalloc.start()

    while n < numero_orbitas:
        for k in tqdm(range(0, total, 1)):
            while cont < Dt_orbita:
                # passo orbital
                Tcond = np.zeros(len(nos))
                Trad = np.zeros(len(nos))
                for i in range(0, len(nos)):
                    for j in range(0, len(nos)):

                        # Preencher o vetor conducao
                        Tcond[j] = (K_m[i, j] * (Temp_ini[j] - Temp_ini[i]))

                        # Preencher o vetor radiacao
                        Trad[j] = (sig * nos[i].e * nos[i].area * fg[i, j] * (
                                Temp_ini[j] ** 4 - Temp_ini[i] ** 4))

                    # Capacitancia termica
                    rho = nos[i].rho
                    cp = nos[i].cp
                    V = nos[i].vol
                    C_i = rho * cp * V

                    # Equacao transiente para cada nó
                    Temp_pos[i] = Temp_ini[i] + (Dt_termico / C_i) * (sum(Tcond) + sum(Trad) +
                                                                      area_rad_ext[i] * calor[k, i] * (nos[i].a)
                                                                      - sig * area_rad_ext[i] * nos[i].e * (Temp_ini_ext[i] ** 4))

                    # Delta os dados da memória
                    del C_i
                    del Tcond
                    del Trad
                    Tcond = np.zeros(len(nos))
                    Trad = np.zeros(len(nos))


                del Temp_ini
                Temp_ini = Temp_pos.copy()
                a = Temp_pos.copy()
                var_temp.append(a)

                for lista_no in Nos_rad:
                    for no in lista_no:
                        j = int(no)
                        Temp_ini_ext[j - 1] = Temp_ini[j - 1]

                del a
                del Temp_pos

                Temp_pos = [300.0 for i in range(0, len(nos))]
                cont += Dt_termico
            cont = 0

        # Elimina os dados da RAM
        nomes = [no.nome for no in nos]
        import csv
        with open(f'Transiente10/beta{beta}/resultado_temp_{n}.csv', 'w', newline='') as arquivo_csv:
            writer = csv.writer(arquivo_csv)
            writer.writerow(nomes)  # Escreve o cabeçalho do CSV

            for lista in var_temp:
                writer.writerow(lista)  # Escreve cada lista como uma linha no CSV

        arquivo_csv.close()  # Fecha o arquivo após a conclusão das operações de escrita
        del var_temp

        var_temp = []
        n += 1


# multiplas orbitas analisadas para um ano com diferentes betas
beta = [0.0, 72.0]
inc = [51.63, 51.63]
for I in range(1, len(beta)):

    calor = pd.read_csv(f'beta{beta[I]}/analise_{inc[I]}/results_{I}/radiacao_{I}/Calor_Incidente.csv', sep=',')

    calor_lump1 = np.array(calor['Total 1'])
    calor_lump2 = np.array(calor['Total 2'])
    calor_lump3 = np.array(calor['Total 3'])
    calor_lump4 = np.array(calor['Total 4'])
    calor_lump5 = np.array(calor['Total 5'])
    calor_lump6 = np.array(calor['Total 6'])

    calor_orbita = np.array([calor_lump1, calor_lump2, calor_lump3, calor_lump4, calor_lump5, calor_lump6])

    calor_orbita = np.transpose(calor_orbita)

    calor2 = np.zeros((len(calor_orbita), len(nos)))

    Nos_rad = list(nos_rad.values())

    k = 0
    for lista_no in Nos_rad:

        for no in lista_no:
            for i in range(0, len(calor)):
                j = int(no)
                calor2[i, j - 1] = calor_orbita[i, k]
        k += 1
    # areas que recebem radiação

    calor = pd.DataFrame(calor2)
    calor.to_csv('calor2.csv')
    area_rad_ext = np.zeros((len(nos)))

    for lista_no in Nos_rad:
        for no in lista_no:
            for comp in nos:
                if no == comp.id:
                    j = int(no)
                    area_rad_ext[j - 1] = comp.kwargs['area_rad']
    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)
    # cria o folder para cada caso
    createFolder(f'Transiente10/beta{beta[I]}')
    lumped(calor2, rad, area_rad_ext, nos, K_m, beta[I])

fim = datetime.now()
var = fim - inicio
from bot_simulacao import bot_temp_trans
tempo = [inicio, fim, var]
bot_temp_trans(tempo)