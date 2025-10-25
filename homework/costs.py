from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from park.entities.ride import Ride
    from park.entities.robot import Robot
    from park.entities.visitor import Visitor


def get_ride_cost(robot: Robot, ride: Ride, picked_visitor: Visitor) -> float:
    """
    Cette fonction calcule le coût pour qu'un robot avec un groupe de visiteurs à bord se rende à une attraction spécifique.
    La distance entre le robot et le début de la file d'attente de l'attraction est utilisée comme coût de base.
    
    On attend que vous amélioriez cette fonction pour qu'elle prenne en compte d'autres facteurs pertinents.
    Le but est bien sûr de générer le plus de cash possible, éviter de créer un embouteillage et maximiser la satisfaction des visiteurs.

    Voici une liste de facteurs que vous pourriez considérer. On ne vous demande pas d'utiliser tous ces facteurs,
    mais de choisir judicieusement ceux qui vous semblent les plus pertinents pour votre stratégie.

    - La vitesse de déplacement du robot: `robot.move_speed`
    - La durabilité restante du robot: `robot.health_percentage` (entre 0 et 100)
    - Le niveau de batterie du robot: `robot.battery_percentage` (entre 0 et 100)
    - La taux de décharge de la batterie par unité de temps: `robot.battery_drain_rate`

    - La satisfaction actuelle des visiteurs: `picked_visitor.satisfaction` (entre 0 et 1)
    - Le nombre d'attractions qu'ils souhaitent visiter au total: `picked_visitor.desired_rides`
    - Le nombre d'attractions visitées: `picked_visitor.completed_rides`

    - La capacité de la file d'entrée de l'attraction, en termes de groupe: `ride.entrance_queue.max_capacity`
    - Le nombre de groupes actuellement dans la file d'entrée de l'attraction: `ride.entrance_queue.count`
    - La capacité de la file d'entrée de l'attraction, en termes de personnes: `ride.entrance_queue.max_members`
    - Le nombre de personnes actuellement dans la file d'entrée de l'attraction: `ride.entrance_queue.member_count`

    - La capacité de la file d'entrée de l'attraction, en termes de groupe: `ride.entrance_queue.max_capacity`
    - Le nombre de groupes actuellement dans la file d'entrée de l'attraction: `ride.entrance_queue.count`
    - La capacité de la file d'entrée de l'attraction, en termes de personnes: `ride.entrance_queue.max_members`
    - Le nombre de personnes actuellement dans la file d'entrée de l'attraction: `ride.entrance_queue.member_count`

    - La capacité de l'attraction: `ride.capacity`
    - La temps que prend l'attraction pour faire un cycle: `ride.duration`
    - Le prix d'entrée de l'attraction, par personne: `ride.entry_price`

    Note importante:
        - Si vous avez déjà implémenté le modèle de reconnaissance des visiteurs, on vous demande d'utiliser
        `robot.predicted_group_type` et `robot.predicted_group_size` pour accéder à la taille et au type du groupe de visiteurs.

        - Si le modèle de reconnaissance n'est pas implémenté, on vous autorise à utiliser
        `picked_visitor.group_type` et `picked_visitor.group_size` à la place.
    """
    # Distance entre le robot et le début de la file d'attente de l'attraction
    distance = robot.transform.position.distance_from(ride.entrance_queue.tail)

    # On pourrait ajouter des facteurs supplémentaires ici pour affiner le coût

    cost = distance  # On utilise la distance comme coût de base
    return cost


def get_visitor_cost(robot: Robot, visitor: Visitor) -> float:
    """
    Cette tâche est optionnelle, mais recommandée si vous souhaitez améliorer encore plus
    la plannification des robots.
    Cette fonction calcule le coût pour qu'un robot aille chercher un groupe de visiteurs spécifique.

    La distance est utilisée comme coût de base. Ceci est fonctionnel, mais on souhaite améliorer
    cette fonction de coût pour qu'elle prenne en compte d'autres facteurs.

    Voici quelques facteurs que vous pourriez considérer:
    - La vitesse de déplacement du robot: `robot.move_speed`
    - La durabilité restante du robot: `robot.health_percentage` (entre 0 et 100)
    - Le niveau de batterie du robot: `robot.battery_percentage` (entre 0 et 100)
    - La taux de décharge de la batterie par unité de temps: `robot.battery_drain_rate`

    - La taille du groupe de visiteurs: `visitor.group_size`
    - Le type du groupe de visiteurs: `visitor.group_type`
    - La satisfaction actuelle des visiteurs: `visitor.satisfaction` (entre 0 et 1)
    - Le nombre d'attractions visitées: `visitor.completed_rides`
    - Le nombre d'attractions qu'ils souhaitent visiter au total: `visitor.desired_rides`

    Par exemple, pour éviter les embouteillages, on peut donner la priorité aux visiteurs
    qui ont finis leur visite et veulent quitter le parc:
        ride_completion_factor = visitor.completed_rides / max(visitor.desired_rides, 1)
        cost = (1 - ride_completion_factor) * distance
    """
    # Distance entre le robot et le visiteur
    distance = robot.transform.position.distance_from(visitor.transform.position)

    # On pourrait ajouter des facteurs supplémentaires ici pour affiner le coût

    cost = distance  # On utilise la distance comme coût de base
    return cost
