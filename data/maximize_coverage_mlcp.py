import pulp
from geopy.distance import geodesic

#formato esperado

# areas = {
#     i: {
#         "coord": (lat, lon),
#         "pop": ...,
#         "saude": ...,
#         "ipvs": ...,
#         "exposicao": ...
#     }
# }

# candidates = {
#     j: (lat, lon)
# }

# existing = {
#     e: (lat, lon)
# }

#dados de entrada vindos do banco

I = list(areas.keys())
J = list(candidates.keys())

#coeficientes

alpha = 1.0
beta = 1.0
gamma = 1.0
delta = 1.0

#pesos das áreas

w = {}
for i in areas:
    data = areas[i]
    w[i] = (
        alpha * data["pop"] +
        beta * data["saude"] +
        gamma * data["ipvs"] +
        delta * data["exposicao"]
    )

#cobertura existente (c_i)

c = {}

#matriz de cobertura (a_ij)

a = {}

Rj = 2.0
Re = 3.0

for i in I:
    a[i] = {}

    for j in J:
        d = geodesic(areas[i]["coord"], candidates[j]).km
        a[i][j] = 1 if d <= Rj else 0

    covered = 0
    for e in existing:
        d = geodesic(areas[i]["coord"], existing[e]).km
        if d <= Re:
            covered = 1
            break

    c[i] = covered

p = 10  #máximo de estações

#modelo

model = pulp.LpProblem("MLCP", pulp.LpMaximize)

#variáveis

x = pulp.LpVariable.dicts("x", J, cat="Binary")
y = pulp.LpVariable.dicts("y", I, cat="Binary")

#objetivo

model += pulp.lpSum(w[i] * y[i] for i in I)

#cobertura

for i in I:
    model += y[i] <= c[i] + pulp.lpSum(a[i][j] * x[j] for j in J)

#limite de estações

model += pulp.lpSum(x[j] for j in J) <= p

#resolver

model.solve()

#resultados

print("Status:", pulp.LpStatus[model.status])

print("\nEstações instaladas:")
for j in J:
    if pulp.value(x[j]) == 1:
        print(f"Candidato {j}")

print("\nÁreas cobertas:")
for i in I:
    if pulp.value(y[i]) == 1:
        print(f"Área {i}")

print("\nValor objetivo:", pulp.value(model.objective))