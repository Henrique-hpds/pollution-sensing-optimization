# pollution-sensing-optimization

## PostGIS via docker

1. Instalar o Docker

Se ainda não tiver, instale o docker.


2. Baixar a imagem do via docker-compose (na raiz do repositório)

```bash
docker compose up -d
```

3. Acessar o terminal do PostGIS

```bash
docker exec -it postgis psql -U postgres
```

4.