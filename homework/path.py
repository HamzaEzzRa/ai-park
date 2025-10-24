from park.internal.math import Vector2D


def astar_g_cost(a: Vector2D, b: Vector2D) -> float:
    """
    Cette fonction devrait calculer le coût g de l'algorithme A* entre deux points dans un espace 2D.
    Le coût g représente le coût actuel pour atteindre un point donné depuis le point de départ.
    Pour le moment, on retourne une valeur constante, ce qui n'est pas utile pour la planification du chemin.

    Vous devez implémenter une heuristique appropriée.
    Référez-vous au tutoriel sur la recherche de chemin et la documentation fournie.
    """
    cost = 0.0  # Valeur constante par défaut
    return cost


def astar_h_cost(a: Vector2D, b: Vector2D) -> float:
    """
    Cette fonction devrait calculer le coût h de l'algorithme A* entre deux points dans un espace 2D.
    Le coût h représente une estimation du coût restant pour atteindre la destination à partir d'un point donné.
    Pour le moment, on retourne une valeur constante, ce qui n'est pas utile pour la planification du chemin.

    Vous devez implémenter une heuristique appropriée.
    Référez-vous au tutoriel sur la recherche de chemin et la documentation fournie.
    """
    cost = 0.0  # Valeur constante par défaut
    return cost
