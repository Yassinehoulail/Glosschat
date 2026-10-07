"""Exceptions de sélection des onglets pour les glossaires KPI.

Chaque clé est le nom du glossaire (tel qu'il est déduit du nom de fichier) et
chaque valeur liste les onglets à utiliser lorsque le classeur ne contient pas
d'onglet nommé ``KPI`` ou ``KPIs``. Les comparaisons ignorent la casse et les
accents.
"""

KPI_SHEETS: dict[str, tuple[str, ...]] = {
    # "nom du glossaire": ("Nom de l'onglet", "Autre onglet"),
    "agrica as": ("Feuille 1",),
    "bp": ("Glossaire",),
    "cimr": ("KPI's",),
    "Inwi Hermès Convention FNTT": ("Convention FNTT",),
    "Inwi Hermès WCB et File Com": ("KPIs WCB et File Com",),
    "red ret": ("Glossaire",),
    "red tlvs": ("Global",),
    "sfr psb": ("Glossaire KPI",),
    "sfr tlv": (
        "CA Fixe",
        "Fixe Dimension",
        "Fixe Mesure",
        "Global",
        "Mobile Dimension",
        "Planification",
        "Prev & Capa",
        "REM Fixe",
        "REM Mobile",
        "Téléphonie Agent",
        "Téléphonie Détails ECH",
        "Téléphonie VDN",
    ),
    "sfrc": ("KPIs V1", "Kpis V2"),
}
