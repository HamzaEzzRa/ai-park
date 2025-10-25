import keras


def get_hidden_layers(x):
    """
    Cette fonction devrait construire et retourner les couches cachées d'un réseau de neurones
    pour la reconnaissance des visiteurs dans le parc d'attractions.

    Le but est de créer une architecture capable d'atteindre une précision haute
    avec le moins de paramètres possible.
    
    Pour le moment, on a une couche cachée qui ne fait rien (Identité) + Dropout qui n'aide pas l'apprentissage toute seule.
    On souhaite que vous remplaciez cela par une ou plusieurs couches que vous jugerez appropriées.
    Vous pouvez utiliser n'importe quelle combinaison des couches suivantes:
        - Conv2D
        - MaxPooling2D
        - Dropout
        - Flatten
        - Dense
    """
    # À remplacer par votre implémentation
    x = keras.layers.Identity()(x)
    x = keras.layers.Dropout(0.9)(x)

    return x
