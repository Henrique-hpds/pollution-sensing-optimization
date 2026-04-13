import pandas as pd
import geopandas as gpd

pd.set_option("display.max_columns", None)

# Importando e tratando os dados considerando a cidade de São Paulo
# Henrique: considerações que eu fiz no processamento
#   - Se a região não tem classificação (i.e., NaN), ipvs é setado como 0
#   - Estou excluindo as estações da CETESB que estão na RMSP, mas não estão na cidade de São Paulo (Osasco, Guarulhos, etc)

# 1. Processamento do IPVS

ipvs = gpd.read_file("raw_data/ipvs_2022/IPVS_2022.shp")
print(ipvs.columns.tolist())
print(ipvs.head(3))

# Seleção de valores e drops de colunas
ipvs = ipvs[(ipvs["NM_MUN"] == "São Paulo")] # Filtra apenas os setores da cidade de São Paulo
ipvs.loc[ipvs["C_IPVS"].isna(), "C_IPVS"] = 0 # Substitui os valores NaN por 0 na coluna C_IPVS
ipvs.drop(columns=["fid", "CD_MUN", "NM_MUN"], inplace=True)
ipvs = ipvs.reset_index(drop=True)

# O sistema de coordenadas do shapefile é diferente do sistema de coordenadas geográficas (latitude e longitude)
# Então precisamos converter as coordenadas para o sistema geográfico para obter as latitudes e longitudes corretas.

# epsg = numero que o claude falou pra usar pra sp EPSG:31983 (SIRGAS 2000 / UTM zone 23S), 
ipvs_proj = ipvs.to_crs(epsg=31983)
# espg 31983 é de metros, nos queremos em latitude e longitude
centroides = ipvs_proj.geometry.centroid.to_crs(epsg=4326)

# Adicionar as colunas de latitude e longitude ao DataFrame
ipvs["latitude"] = centroides.y # Y = Latitude
ipvs["longitude"] = centroides.x # X = Longitude

# print(ipvs.shape)       # quantas linhas e colunas tem
# print(ipvs.head())      # primeiras 5 linhas


# 2. Processamento das UBSs

# Extrair dados do Cadastro Nacional de Estabelecimentos de Saúde (CNES)
cnes = pd.read_csv('raw_data/dados_cnes/tbEstabelecimento202602.csv', sep=';' ,encoding='latin-1', low_memory=False)

# Filtrar apenas as UBSs da cidade de São Paulo
ubs = cnes[(cnes["TP_UNIDADE"] == 2) & (cnes["CO_MUNICIPIO_GESTOR"] == 355030) & (cnes["NO_RAZAO_SOCIAL"] == "PREFEITURA DO MUNICIPIO DE SAO PAULO") & (cnes["CO_MOTIVO_DESAB"].isna()) & (cnes["TP_GESTAO"] == "M")] # Filtra apenas as linhas onde TP_UNIDADE é 2 (UBS), CO_MUNICIPIO_GESTOR é 355030 (São Paulo), NO_RAZAO_SOCIAL é "PREFEITURA DO MUNICIPIO DE SAO PAULO", CO_MOTIVO_DESAB é NaN (não desativada), e TP_GESTAO é "M" (municipal)

considerar_nulos_internet = True

if considerar_nulos_internet:
    ubs = ubs[ubs["ST_CONEXAO_INTERNET"] != "N"] # Filtra apenas as linhas onde ST_CONEXAO_INTERNET não é NaN
else:
    ubs = ubs[ubs["ST_CONEXAO_INTERNET"] == "S"] # Filtra apenas as linhas onde ST_CONEXAO_INTERNET é "S"

# Drops de colunas
ubs.drop(columns=["NU_CNPJ_MANTENEDORA", "TP_PFPJ", "NIVEL_DEP", "NO_RAZAO_SOCIAL", "NO_LOGRADOURO", "NU_ENDERECO", "NO_COMPLEMENTO", "CO_CEP", "CO_REGIAO_SAUDE", "CO_MICRO_REGIAO", "CO_DISTRITO_SANITARIO", "CO_DISTRITO_ADMINISTRATIVO", "NU_TELEFONE", "NU_FAX", "NO_EMAIL", "NU_CPF", "NU_CNPJ", "CO_ATIVIDADE", "CO_CLIENTELA", "NU_ALVARA", "DT_EXPEDICAO", "TP_ORGAO_EXPEDIDOR", "DT_VAL_LIC_SANI", "TP_LIC_SANI", "TP_UNIDADE", "CO_TURNO_ATENDIMENTO", "CO_ESTADO_GESTOR", "CO_MUNICIPIO_GESTOR", "TO_CHAR(DT_ATUALIZACAO,'DD/MM/YYYY')", "CO_USUARIO", "CO_CPFDIRETORCLN", "REG_DIRETORCLN", "ST_ADESAO_FILANTROP", "CO_MOTIVO_DESAB", "NO_URL", "TO_CHAR(DT_ATU_GEO,'DD/MM/YYYY')", "NO_USUARIO_GEO", "CO_NATUREZA_JUR", "TP_ESTAB_SEMPRE_ABERTO", "ST_GERACREDITO_GERENTE_SGIF", "ST_CONEXAO_INTERNET", "CO_TIPO_UNIDADE", "NO_FANTASIA_ABREV", "TP_GESTAO", "TO_CHAR(DT_ATUALIZACAO_ORIGEM,'DD/MM/YYYY')", "CO_TIPO_ESTABELECIMENTO", "CO_ATIVIDADE_PRINCIPAL", "CO_TIPO_ABRANGENCIA", "ST_COWORKING", "ST_CONTRATO_FORMALIZADO"], inplace=True)
ubs = ubs.reset_index(drop=True)

# 3. Processamento das estações da CETESB

# Extrair dados das estações de monitoramento da qualidade do ar da CETESB
cetesb = pd.read_excel("raw_data/postos-de-cetesb.xlsx")

# Filtrar apenas as estações da cidade de São Paulo
cetesb = cetesb[(cetesb["Município"] == "São Paulo")]

cetesb.drop(columns=["UF", "Município", "COD", "Observações"], inplace=True) # não lembro oq significava Automático x Manual, tirar dps se necessário
cetesb = cetesb.replace({"NÃO": 0, "SIM": 1})

cetesb = cetesb.reset_index(drop=True)

# As coordenadas da CETESB estão em UTM 23S (SIRGAS 2000); convertemos para graus decimais (WGS84)
cetesb = gpd.GeoDataFrame(
    cetesb,
    geometry=gpd.points_from_xy(
        pd.to_numeric(cetesb["LONGITUDE"], errors="coerce"),
        pd.to_numeric(cetesb["LATITUDE"], errors="coerce"),
    ),
    crs="EPSG:31983",
)
cetesb = cetesb.to_crs(epsg=4326)
cetesb["LATITUDE"] = cetesb.geometry.y
cetesb["LONGITUDE"] = cetesb.geometry.x
cetesb.drop(columns="geometry", inplace=True)
    
print(cetesb.head())