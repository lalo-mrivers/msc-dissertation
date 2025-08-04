from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# Conexión al servidor Qdrant
client = QdrantClient(
    host="uhtred.inf.ed.ac.uk",  # Cambia si usas otro host
    port=6333          # Cambia si usas otro puerto
)

# Opcional: ver las colecciones actuales
# Listar colecciones actuales
def listar_collecciones(pref=None):
    collections = client.get_collections().collections
    print("Colecciones existentes:")
    for col in collections:
        if pref and not col.name.startswith(pref):
            continue
        collection_name = col.name
        stats = client.get_collection(collection_name=collection_name)
        num_points = stats.points_count  # o stats.points_count en versiones recientes
        print(f"Colección: {collection_name}, Número de puntos: {num_points}")

# Borrar una colección
def borrar_coleccion(nombre):
    try:
        client.delete_collection(collection_name=nombre)
        print(f"Colección '{nombre}' eliminada.")
    except UnexpectedResponse as e:
        print(f"Error al eliminar la colección '{nombre}': {e}")

# Borrar múltiples colecciones
def borrar_multiples(colecciones):
    for nombre in colecciones:
        borrar_coleccion(nombre)

# ---------- Ejemplo de uso ----------
if __name__ == "__main__":
    listar_collecciones('med')

    # Lista de colecciones a borrar
    #colecciones_a_borrar = ['medical_dpr_collection']#["medical_dpr_collection", "medical_pubmedbert_collection", "medical_modern_bert_collection"]
    #borrar_multiples(colecciones_a_borrar)
