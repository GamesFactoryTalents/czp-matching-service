"""
Reference taxonomy from the Careers Zone app.
Used to give Claude precise context about valid values for structured fields.
"""

CATEGORIES = [
    "Art & Animation",
    "Audio & Sound",
    "Business & Management",
    "Data & Analytics",
    "Game Design",
    "Localisation",
    "Monetisation",
    "Player Support & Community",
    "Product & LiveOps",
    "Production",
    "Programming & Engineering",
    "QA & Testing",
    "UA & Marketing",
    "UI & UX Design",
    "Writing",
]

PLATFORMS = [
    "PC",
    "Console",
    "Mobile",
    "VR",
    "AR",
    "Web",
    "Social",
    "Not applicable",
]

ENGINES = [
    "Unity",
    "Unreal Engine (UE)",
    "Godot",
    "GameMaker: Studio",
    "Cocos2d",
    "Construct",
    "CryENGINE",
    "Frostbite",
    "Amazon Lumberyard",
    "ARkit",
    "Blender",
    "Corona SDK",
    "Havok",
    "libGDX",
    "Phaser",
    "PlayCanvas",
    "RPG Maker VX Ace",
    "Turbulenz",
    "LÖVE",
    "RAGE Engine",
    "ID Tech",
    "Infinity Engine",
    "HeroEngine",
    "Kivy",
    "Havok Vision Engine",
    "GameSalad",
    "The Dark Engine",
    "Consolution",
]

GENRES = [
    "Action",
    "Adventure",
    "Arcade",
    "AR/Location Based",
    "Board Games",
    "Card Games",
    "Casino",
    "Casual",
    "Driving",
    "Educational",
    "Fighting",
    "Hyper-casual",
    "Kids Games",
    "Life-style",
    "Match 3",
    "MMO",
    "Music",
    "Platformer",
    "Puzzle",
    "Racing",
    "RPG",
    "Shooter",
    "Simulation",
    "Sports",
    "Strategy",
    "Word Games",
]


def taxonomy_context() -> str:
    """Returns a compact taxonomy reference for inclusion in prompts."""
    return f"""
GAMES INDUSTRY TAXONOMY (values used in this system):
- Categories: {', '.join(CATEGORIES)}
- Platforms: {', '.join(PLATFORMS)}
- Engines: {', '.join(ENGINES)}
- Genres: {', '.join(GENRES)}
""".strip()
