from __future__ import annotations

from enum import Enum, IntFlag

from ai.fsm.core import Machine


class RobotState(Enum):
    ROAMING = "roaming"
    PICK_VISITOR = "pick_visitor"
    PICK_RIDE = "pick_ride"
    CHARGING = "charging"


class RobotTrigger(IntFlag):
    VISITOR_IN_QUEUE = 1 << 0
    VISITOR_ON_BOARD = 1 << 1
    VISITOR_DROP_OFF = 1 << 2
    LOW_BATTERY = 1 << 3
    FULL_BATTERY = 1 << 4


def get_robot_fsm() -> Machine:
    """
    Cette fonction devrait construire et retourner la machine à états finis (FSM)
    pour un robot du parc d'attractions.
    
    On vous fournit ci-dessus les états et les entrées (triggers) nécessaires.
    On attend que vous créiez une instance de Machine avec les états `RobotState` et l'état initial `ROAMING`.
    Vous devez ensuite ajouter les transitions entre les états en fonction des entrées dans `RobotTrigger`.
    
    Si vous trouvez des difficultés, on vous encourage à revoir le tutoriel sur les automates finis
    et consulter la documentation fournie.

    Notes concernant l'énumération `RobotTrigger`:
        C'est une énumération spéciale où chaque entrée est une puissance de deux.
        Autrement dit, `1 << n` est équivalent à `2 à la puissance n`.
        Ce type d'entrée est souvent appelé masque binaire (bitmask).
        
        Cela est utile pour combiner plusieurs entrées en une seule valeur unique en utilisant des opérations binaires.
        Par exemple, pour combiner `VISITOR_IN_QUEUE` et `LOW_BATTERY`, vous pouvez faire:
            RobotTrigger.VISITOR_IN_QUEUE | RobotTrigger.LOW_BATTERY

        L'opération `|` (OU binaire) combine les bits de deux entrées.
            `1 << 0` => 1 => 0001 (en binaire)
            `1 << 3` => 8 => 1000 (en binaire)
            ----------------------------------
            1 | 8 => 9 => 1001 (en binaire)
        Si on considère que les puissances de deux, la valeur 9 (1001 en binaire) ne peut être obtenue
        que par la combinaison de 1 et 8.
        On est ainsi capable de vérifier facilement quelles entrées sont actives dans une valeur combinée.
    """

    fsm = Machine()  # Remplacez par une instance correcte de Machine avec les états et l'état initial
    # Ajoutez les transitions entre les états après.

    return fsm
