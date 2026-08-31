"""How many posts a hashtag actually carries, and which band that puts it in.

Step 6 asks Gemini for "broad", "mid" and "niche" tags. Those tiers describe
how the *words* feel, not how many posts the tag carries, and the two come
apart badly at the long-tail end: #Piezophile and #Osmolyte are niche in the
vocabulary sense and empty in the only sense that matters. A tag with a few
hundred lifetime posts has no browsers, so it is not reach - it is a character
count. At the other end #MarineBiology has millions, and a small account never
surfaces in it.

This module puts a number against a tag and sorts it into a band:

    dead       below TAG_VOLUME_FLOOR         - dropped
    niche      floor .. TAG_VOLUME_NICHE_MAX  - small enough to rank in
    mid        .. TAG_VOLUME_MID_MAX          - the band that actually converts
    broad      .. TAG_VOLUME_BROAD_MAX        - places the post in a subject
    saturated  above that                     - capped, never more than a token

Where the number comes from, best first:

  1. `measured` - a count in the cache file (TAG_VOLUME_FILE), which is also
     where a provider's answers are kept. Measured counts always win.
  2. `provider` - an optional callable named by TAG_VOLUME_PROVIDER as a dotted
     path, for whoever has a data source. Answers are cached with a TTL.
  3. `estimate` - the offline fallback, always available. No key, no network.

There is deliberately no built-in Instagram provider. The IG Graph API hashtag
node exposes `id` and `name` and no post count, and it caps an account at 30
unique hashtag lookups per rolling 7 days - less than one week of posts at
POST_HASHTAG_COUNT tags each. It cannot answer this question.

The estimator is a rarity model, not an oracle. It reads a tag as the words it
is built from and prices it on how ordinary those words are and how many of
them there are. It reliably separates a tag made of everyday words
(#DeepSeaFish, #LavaTubes) from one built on a term almost nobody types
(#Piezophile, #MariusHills), which is the failure this exists to stop. It is
coarse in the middle - #Snailfish reads larger than it is - so it is used to
reject and to rank, never reported as fact: every estimated figure is labelled
`estimate` in post.json so nothing downstream mistakes it for a measurement.

Standalone use, to see what it makes of a set of tags:
    python pipeline/tagvolume.py Piezophile DeepSeaFish OceanLife Snailfish
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

#: Bands, smallest first. The thresholds themselves live in config.py.
BANDS = ("dead", "niche", "mid", "broad", "saturated")

#: Bands a tag may be used from at all.
USABLE = ("niche", "mid", "broad", "saturated")

#: Which band each of step 6's tiers wants, in order of preference. A tier
#: falls back down this list rather than going empty.
TIER_BANDS = {
    "broad": ("broad", "saturated", "mid"),
    "mid": ("mid", "niche", "broad"),
    "niche": ("niche", "mid", "broad"),
}


class VolumeError(RuntimeError):
    """Raised when TAG_VOLUME_PROVIDER names something unusable."""


# ---------------------------------------------------------------------------
# The estimator's vocabulary
#
# Two buckets: words almost every caption uses, and ordinary vocabulary. A word
# in neither - a Latin binomial, a technical term, a place nobody has heard of -
# is what sinks a tag, so the penalty for it is deliberately severe.
# ---------------------------------------------------------------------------

_COMMON = """
a about above across act add after again against age ago air all almost alone
along already also always among an and animal another answer any appear are
area around as ask at away back bad bag ball band bank base be beach bear
beautiful because become bed been before begin behind being below best better
between big bird bit black blood blue board boat body book both bottom box boy
brain break bring brother build building burn but buy by call came can car
care carry case cat catch cause cell center centre century certain chance
change check child city class clean clear close cloud cold color colour come
common company complete computer control cook cool copy could count country
couple course cover create cross cry cup current cut dark data day dead deal
death deep design detail develop did die difference different dinner direct do
doctor dog door double down draw dream drink drive drop dry during each early
earth east easy eat edge effect egg eight either else end energy enough enter
equal even evening event ever every example eye face fact fall family far fast
father fear feed feel feet few field fight figure fill film final find fine
finger finish fire first fish five fit floor flow flower fly follow food foot
for force forest form found four free friend from front full fun future game
garden gas gave general get girl give glass go gold gone good got great green
ground group grow guess hair half hand happen hard has hat have he head hear
heart heat heavy held help her here hide high hill him his hit hold hole home
hope horse hot hour house how huge human hundred hunt ice idea if image
important in inch include increase inside into iron is island it its job join
jump just keep key kill kind king know known lack lake land language large
last late later laugh law lay lead learn least leave left leg length less let
letter level lie life lift light like line list listen little live local long
look lose lost lot love low machine made main major make man many map mark
market mass master match material may maybe me mean measure meat meet member
memory men metal method middle might mile milk million mind mine minute miss
mix model modern moment money month moon more morning most mother mountain
mouth move much music must my name nation natural nature near need never new
news next night nine no north not note nothing now number object ocean of off
offer office often oil old on once one only open or order other our out over
own page pain paint pair paper part pass past path pattern pay people perfect
perhaps period person phone photo pick picture piece place plan plane plant
play please point poor position possible power practice prepare present press
pretty price print probably problem process produce product program property
protect prove public pull push put question quick quiet quite race radio rain
raise range rate rather reach read ready real reason receive record red reduce
remain remember remove repeat report rest result return rich ride right ring
rise river road rock roll room root round row rule run safe said sail salt
same sand save saw say scale school science sea search season seat second see
seed seem sell send sense sent serve set seven several shape share she ship
shoe shop short should show side sign silver simple since sing single sister
sit six size skin sky sleep slow small smell smile smoke snow so soft soil
soldier solid some son song soon sort sound south space speak special speed
spend spot spread spring square stand star start state stay step stick still
stone stop store storm story straight strange street strong study subject such
sudden sugar summer sun supply support sure surface surprise sweet swim system
table tail take talk tall teach team tell ten term test than that the their
them then there these they thick thin thing think third this those though
thought thousand three through throw thus time tiny to today together told
tomorrow tone too took tool top total touch toward town track trade train
travel tree trip trouble true try turn twice two type under understand unit
until up upon us use usual valley value very view village visit voice wait
walk wall want war warm was watch water wave way we wear weather week weight
well went were west what wheel when where which while white who whole why wide
wife wild will win wind window wing winter wire wish with within without woman
wonder wood word work world would write wrong yard year yellow yes yet you
young your
"""

_ORDINARY = """
absorb abyss acid active adapt advance adventure ancient angle angler apparent
archive arctic art artist atmosphere atom author average balance barnacle
beast behaviour belief bell birth blast bloom bone border boundary branch
breath breed bridge bright broad bubble budget bug bury cable calm camera camp
canyon capture carbon career cargo cave celebrate chain challenge chamber
channel chaos character charge chart chemical chief circle circuit claim clam
classic cliff climate climb clock cloth coast coastline coat cobra code coffee
collapse collect colony column combine comfort command comment community
compare compete complex concept concern concrete condition conduct connect
consider constant consume contact contain content contrast convert copepod
core corner cost crab crack craft crash crater cream creature crew crime
crisis critical crocodile crop crowd crystal culture curious curve custom
cuttle cycle damage dance danger daylight debate decade decide declare decline
defence define degree delay deliver demand dense depend depth desert desire
destroy detect device diet digital dimension disaster discover disease display
distance district divide document dolphin domain dominant downstream drift
drug dust duty eagle earn echo ecology economy edit educate eel effort elect
electric element emerge emotion empire employ empty enable engine engineer
enhance enormous ensure entire entry environment episode equipment error
escape essential estimate evidence evolve exact examine exceed exchange excite
exist exit expand expect expert explain explore expose extend extreme fabric
factor factory fade fail faint falcon fame famous fashion fatal fault favour
feature federal feedback festival fever fiction file filter finance firefly
flame flamingo flash flat fleet flight float flood fluid focus fog forecast
formal format former fossil foundation fragile frame freeze frequency fresh
friction frozen fuel function fund fungus furniture gain galaxy gallery gap
gather gear gene generate gentle genuine gesture giant glacier glance globe
glow grain grand grant graph grass gravel gravity graze greet grid grip guard
guide gull habit habitat hall handle harbour harvest hazard health heaven
height helmet heritage heron hidden hint historic history hollow honest honour
hormone host hunger hybrid hydrogen hypothesis icon identify ignore illness
illusion imagine impact impress improve impulse incident income indicate
industry infant infect inform initial injury ink innocent input insect insight
inspect install instance instant instead instinct institute instrument
insulate intense interior internal internet interval invent invest invisible
involve isolate isopod issue jacket jaw jelly jewel journal journey judge
juice jungle justice keen kernel kingdom kitchen knee knife knot knowledge
krill label laboratory labour ladder lamp landmass landscape lantern laser
latitude launch lava layer leaf league leather lecture legend leisure lemon
lens lesson liberty library licence limb limestone limit link liquid
literature litter lizard load loan lobby lobster locate lock lodge logic
lonely loop loose lord loud loyal lucky lung luxury magic magnet mammal manage
manner manta mantle manual marble march margin marine marsh mask massive
matrix matter mature maximum meadow meaning mechanic medal media medical
medicine medium melt mental mention menu mercury mercy merge mesh message
meteor midnight migrate mild military mineral minimum mining minor miracle
mirror missile mission mist mixture mobile mode moisture molecule monitor
monkey monster monument mood moonlight moral motion motor mould mount muscle
museum mushroom mussel mutual mystery myth naked narrow native nautilus navy
neat neck negative neighbour nerve nest network neutral newt niche nightfall
noble noise normal nose notice novel nuclear nurse nutrient oak obey observe
obtain obvious occupy occur octopus odd odour offend official offset offshore
onion opera operate opinion oppose optical option orbit orca orchard organ
organic origin ornament osprey otter outcome outdoor outer outline output oval
oven overcome owl oxygen oyster ozone pace pack packet palace pale palm panel
panic panther parade parallel parent park parrot partial particle partner
passage passenger passion patch patent patient patrol pause peace peak pearl
peculiar pedal peel pelican penalty pencil penguin pension pepper perceive
percent perform permit persist petal petrol phase phenomenon philosophy phrase
physical physics piano pierce pigeon pigment pilot pine pink pioneer pipe
pirate pitch pixel planet plankton plasma plastic plate plateau platform
pleasure pledge plenty plot plug plunge poet poison polar pole policy polish
politics pollen pollute pond popular porpoise port portion portrait positive
possess post potential pottery pound pour poverty powder praise precise
predator predict prefer pregnant premium presence preserve pressure prevent
previous prey primary prime primitive principle prison private privilege prize
probe proceed profile profit project promise promote prompt proof proper
propose prospect protein protest proud province publish puffin pulse pump
pupil purchase pure purple purpose pursue puzzle pyramid quality quantity
quarry quarter queen quest queue quote rabbit rack radar radiate radical
radius raft rail rainbow rainfall rally random rank rapid rare ratio raven raw
ray razor react realm rebel recall receipt recent recipe recover recruit
recycle reef refer refine reflect reform refuge refuse regard region register
regret regular reject relate relax release relief religion reluctant rely
remark remote render repair replace reply republic reputation request require
rescue research reserve reside resist resolve resort resource respect respond
response restore restrict retail retain retire retreat reveal revenue reverse
review revive reward rhythm rib ribbon ridge rifle rigid riot risk ritual
rival riverbed roar roast robot rocket rod rodent romance roof rope rose rot
rough route routine royal rubber ruin rural rush rust sacred sacrifice saddle
safety sailor salamander salmon sample sanctuary sandstone satellite satisfy
savage scallop scan scandal scar scarce scatter scene scent scheme scholar
scope score scout scrap scratch scream screen screw script sculpture seabed
seahorse seal seashore secret section sector secure sediment segment seize
select senior sensor sentence separate sequence series serious servant session
settle severe shade shadow shaft shallow shame shark sharp shed sheep sheer
sheet shelf shell shelter shield shift shine shock shoot shore shoreline
shortage shoulder shout shower shrimp shrink shrub sight signal silence silk
similar simulate sincere sink siren site situate skeleton sketch ski skill
skull slab slave sleeve slice slide slight slip slope slot slug smart smash
smooth snail snake snap sniff snowfall soak soap social society sodium solar
sole solution solve soul source sovereign span spare spark sparrow spatial
spawn species specific specimen spectrum speech sphere spice spider spike
spill spin spine spiral spirit splash split sponge spoon sport spray spur spy
squad squeeze squid stable stack stadium staff stage stain stake stall stamp
standard staple stark starlight starling statue status steady steal steam
steel steep steer stem sterile stiff stimulus sting stingray stir stitch stock
stomach storage strain strait strand strategy stream strength stress stretch
strict strike string strip stripe stroke structure struggle stubborn studio
stumble stun sturdy sturgeon style submarine submit subtle suburb succeed
suffer sufficient suggest suit sulphur summit sunlight sunrise sunset superior
supreme surge surgeon surplus surround survey survive suspect suspend sustain
swallow swamp swap swarm sway sweep swell swift swing switch sword symbol
symptom tackle tactic tag talent tank tape target task taste tax teeth
telescope temper temple tempt tenant tender tension tent tentacle terminal
terrain terrible territory terror texture theatre theme theory therapy thermal
thesis thief thorn thread threat thrill thrive throat throne thrust thumb
thunder ticket tidal tide tie tiger tight tile timber tissue title toad toast
tobacco toe token tolerate toll tomb tongue tonne torch tornado tortoise
torture toss tough tour tournament tower toxic toy trace tractor tradition
traffic tragedy trail trait transfer transform transit translate transmit
transport trap trash trauma tray treasure treat treaty tremble trench trend
trial triangle tribe tribute trick trigger trim triumph troop tropical truck
trunk trust truth tube tuck tumble tuna tunnel turbine turtle tusk tutor
twilight twin twist typical tyre ultimate umbrella unable uncle uncover
underground undersea uniform unify unique unite universe unknown unusual
update upgrade upper upset upstream urban urchin urge urgent usage utility
vacant vacuum vague valid valve vanish vapour variety various vast vault
vegetable vehicle vein velvet vendor venture venue verdict verify verse
version vertical vessel veteran viable vibrate victim victory video vigour
villa vine vinegar violence violet viral virtual virus visible vision visual
vital vitamin vivid vocal volcano volume volunteer vote voyage vulnerable
vulture wage wagon waist walrus wander warehouse warfare warn warrant warrior
waste waterfall watershed wax weak wealth weapon weave web wedge weed weird
welcome welfare wetland whale wheat whisper whistle wilderness wildfire
wildlife willow windmill wisdom witness wolf womb wool worm worry worship
worth wound wrap wreck wrist yacht yield yoga zone zoo
"""

#: Subject vocabulary: real words, rarer in a caption than "water" but nothing
#: like a coinage. They make a tag smaller without making it empty, which is
#: exactly the long-tail band worth having. Without this bucket #MarineBiology
#: reads the same as #Piezophile, and the filter throws away the good tags.
_TECHNICAL = """
abyssal acoustic aerodynamic aerosol algae alkaline alloy alluvial amber amino
ampere amphibian anatomy anomaly antenna anthropology antibody aperture
aquatic aquifer arachnid archaeology arthropod asteroid astronaut astronomy
atmospheric aurora avalanche bacteria basalt bedrock benthic binary biology
biomass biome biosphere botany buoyancy caldera calorie canopy capillary
carnivore cartography catalyst cellular cetacean chemistry chlorophyll
chromosome circadian coastal cognition colloid combustion condensation
conductivity conifer conservation constellation continental convection coral
corrosion cosmic cosmology crustacean crustal cryogenic cyclone deciduous
decompose density deposit desalination diffusion digestion dinosaur dioxide
dormant drought dynamo earthquake echolocation eclipse ecosystem
electromagnetic electron elevation embryo emission endemic entropy enzyme
epicentre equator equinox erosion eruption estuary evaporation evolution
excavation exoplanet extinct fauna fermentation fertile filament fission fjord
flora fluorescent forage foraging fossilise fracture frontal fusion galactic
gaseous gastropod genetic genome geologic geology geometry geothermal
germination glacial gland gradient gravitational greenhouse groundwater gyre
habitation halite helium hemisphere herbivore hibernate hydraulic hydrothermal
igneous immune incubate inertia infrared insulation invertebrate ion
ionosphere irrigation isotope kinetic lagoon landslide larva lichen lifecycle
luminescence luminescent luminous lunar magma magnetic magnetosphere mammalian
metabolic metabolism metamorphic meteorite methane microbe microbial
microscope migration mineralogy molecular mollusc monsoon moraine morphology
mutation nebula neuron neutron nitrogen nocturnal observatory oceanic
oceanography optics orbital organism ornithology oscillation osmosis
paleontology parasite pathogen permafrost photosynthesis physiology
phytoplankton plumage pollination polymer population precipitation predation
prehistoric primate propulsion proton pulsar quantum quasar radiation
radioactive rainforest reactor reflective refraction regolith relativity
reptile reservoir resonance respiration rotation saline salinity sandbar
savanna seafloor sedimentary seismic semiconductor sensory silicon skeletal
solstice sonar spawning spectral spectrometer stalactite stratosphere
subduction submersible substrate subsurface supernova surveying symbiosis
synthesis taxonomy tectonic telemetry temperate terrestrial thermodynamics
topography trajectory transpiration tropic tsunami tundra turbulence
ultraviolet vaccine vascular velocity ventilation vertebrate viscosity
volcanic voltage wavelength weathering zoology zooplankton
"""

#: Names the subject matter keeps using. A wordlist cannot hold every proper
#: noun, and it does not need to - but a bucket of the ones this kind of
#: content leans on is the difference between #RomanEmpire reading as the very
#: large tag it is and reading as a coinage. Priced as ordinary vocabulary:
#: a well-known name carries a tag about as well as a well-known word.
_NAMES = """
africa african alaska alexandria amazon america american andes antarctic
antarctica arabia arabian asia asian athens atlantic atlantis australia
australian aztec babylon baltic bengal bering brazil brazilian britain british
byzantine cairo california cambodia canada canadian caribbean caspian celtic
china chinese colombia congo cretaceous crete cuba cyprus danube delhi denmark
egypt egyptian england english ethiopia europe european everest france french
galapagos ganges german germany ghana gobi greece greek greenland hawaii
hawaiian himalaya himalayan iceland inca incan india indian indonesia indus
iran iraq ireland irish israel italian italy jamaica japan japanese java
jordan jupiter jurassic kenya korea korean lebanon libya london madagascar
malaysia mali maya mayan mediterranean mekong mesopotamia mexican mexico
mongol mongolia morocco moscow nepal netherlands newzealand niger nigeria nile
nordic norse norway norwegian olympus ottoman pacific pakistan palestine
panama paris patagonia persia persian peru peruvian philippines phoenician
pluto poland polynesia pompeii portugal pyrenees rhine roman rome russia
russian sahara saturn scandinavia scotland scottish serengeti siberia siberian
sicily sinai singapore somalia spain spanish sparta sumatra sumerian sweden
swiss switzerland syria taiwan tanzania thailand tibet tibetan tokyo triassic
tunisia turkey uganda ukraine uranus venezuela venice venus victoria vietnam
viking wales welsh yangtze yemen yucatan zambia zimbabwe
"""

#: Suffixes worth stripping before giving up on a word. "oceanic" is not in the
#: lists above; "ocean" is, and the tag is about the same size either way.
#: Longest first, so "ology" is tried before "y".
_SUFFIXES = (
    "ography", "ically", "ology", "ation", "ition", "ement", "ness", "ship",
    "able", "ible", "ical", "ance", "ence", "ious", "eous", "ity", "ing",
    "ers", "est", "ies", "ive", "ise", "ize", "ial", "ian", "ic", "al", "ed",
    "er", "es", "ly", "y", "s",
)

#: Prefixes that build a real term out of a real root. Without these
#: #Bioluminescence reads as a coinage and gets thrown away, when it is one of
#: the larger tags in this subject. A prefixed word is priced as subject
#: vocabulary, not as an unknown.
_PREFIXES = (
    "electro", "magneto", "astro", "cyber", "hydro", "micro", "macro",
    "multi", "neuro", "photo", "psycho", "super", "therm", "thermo", "trans",
    "ultra", "aero", "anti", "auto", "endo", "hyper", "inter", "intra",
    "mega", "nano", "para", "poly", "proto", "pseudo", "radio", "semi",
    "tele", "bio", "eco", "exo", "geo", "iso", "mono", "sub",
)

COMMON_WORDS = frozenset(_COMMON.split())
ORDINARY_WORDS = frozenset(_ORDINARY.split())
TECHNICAL_WORDS = frozenset(_TECHNICAL.split())
NAME_WORDS = frozenset(_NAMES.split())

#: What each rarity rank costs, in orders of magnitude.
#:   0  everyday word                                  #Ocean, #Fish
#:   1  ordinary vocabulary                            #Habitat, #Volcano
#:   2  subject vocabulary, or recognisable once a
#:      suffix comes off                               #Marine, #Oceanic
#:   3  not a word anyone types                        #Piezophile, #Marius
_RANK_COST = {0: 0.0, 1: 0.35, 2: 0.9, 3: 3.2}

#: An unknown word next to a real one only makes a tag small - #MariusHills
#: has "hills" holding it up. A tag with nothing recognisable in it at all is
#: a coinage, and coinages are empty. That is two different failures, so the
#: second costs extra rather than the per-word penalty carrying both.
_ALL_UNKNOWN_COST = 2.0

#: Where a one-word tag of the commonest kind - #Ocean, #Space - starts.
_BASE_LOG = 7.4

#: What each word past the first costs. Compounds get small fast.
_WORD_COST = 1.1

#: Tag length past this is its own signal, whatever the words are.
_LONG_TAG_CHARS = 22
_LONG_TAG_COST = 0.5

_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")
_WORDISH_RE = re.compile(r"[A-Za-z]+")


# ---------------------------------------------------------------------------
# Reading a tag as words
# ---------------------------------------------------------------------------

def segment(tag: str) -> list[str]:
    """Split a tag into lowercase words.

    CamelCase does the work when the model supplies it - #DeepSeaFish splits
    cleanly. An all-lowercase tag is segmented greedily against the vocabulary
    instead, longest match first, so #deepseafish still comes apart.
    """
    text = str(tag or "").strip().lstrip("#")
    if not text:
        return []

    parts = [p.lower() for p in _CAMEL_RE.findall(text)]
    if len(parts) > 1:
        return parts

    return _greedy_split(parts[0] if parts else text.lower())


def _greedy_split(word: str) -> list[str]:
    """Break a run-together tag into words, longest first.

    A word recognised whole - directly, or once a suffix or prefix comes off -
    is left alone, or "photography" comes apart into photo+graphy and is
    priced as two words, one of them nonsense.

    Every piece of a split has to be a word. Taking the longest known head and
    keeping whatever is left turns "anglerfish" into angle+rfish and calls a
    perfectly ordinary tag a coinage, so a split that cannot account for the
    whole word is refused and the word is judged as it stands.
    """
    if not word:
        return []
    if _recognised(word):                    # whole thing is a word: done
        return [word]
    return _decompose(word) or [word]


def _decompose(word: str, depth: int = 0) -> list[str] | None:
    """Split `word` into known pieces, all the way to the end, or give up."""
    if not word:
        return []
    if depth >= 4:
        return None
    for size in range(len(word), 2, -1):
        head = word[:size]
        if not _known(head):
            continue
        if size == len(word):
            return [head]
        tail = _decompose(word[size:], depth + 1)
        if tail is not None:
            return [head, *tail]
    return None


def _recognised(word: str) -> bool:
    """Known outright, or reducible to something known."""
    return bool(_known(word) or _destem(word) or _deprefix(word))


def _deprefix(word: str, depth: int = 0) -> str | None:
    """The root `word` is built on, if a known prefix comes off cleanly.

    Prefixes stack - "astrophotography" is astro + photo + graph - so this
    recurses a couple of levels rather than giving up after the first.
    """
    if depth >= 2:
        return None
    for prefix in _PREFIXES:
        if len(word) > len(prefix) + 3 and word.startswith(prefix):
            rest = word[len(prefix):]
            if _known(rest) or _destem(rest) or _deprefix(rest, depth + 1):
                return rest
    return None


def _known(word: str) -> bool:
    return (word in COMMON_WORDS or word in ORDINARY_WORDS
            or word in TECHNICAL_WORDS or word in NAME_WORDS)


def _destem(word: str) -> str | None:
    """The listed word `word` is built from, if a suffix comes off cleanly."""
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            stem = word[: -len(suffix)]
            for candidate in (stem, stem + "e", stem[:-1] if stem else ""):
                if candidate and _known(candidate):
                    return candidate
    return None


def rarity(word: str) -> int:
    """Rank a single word 0 (everyday) to 3 (nobody types this)."""
    low = str(word or "").lower()
    if not low or not _WORDISH_RE.fullmatch(low):
        return 3
    if low in COMMON_WORDS:
        return 0
    if low in ORDINARY_WORDS or low in NAME_WORDS:
        return 1
    if low in TECHNICAL_WORDS:
        return 2
    return 2 if (_destem(low) or _deprefix(low)) else 3


# ---------------------------------------------------------------------------
# The estimate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Volume:
    """What is known about one tag."""

    tag: str
    count: int
    source: str                              # measured | provider | estimate
    band: str

    @property
    def measured(self) -> bool:
        return self.source != "estimate"

    def as_dict(self) -> dict:
        return {"count": self.count, "source": self.source, "band": self.band}


def estimate(tag: str) -> int:
    """A rough post count for `tag`, from how ordinary its words are.

    Calibrated against tags of known size: #Ocean lands in the tens of
    millions, #DeepSeaFish in the low hundreds of thousands, #Piezophile in
    the hundreds. The middle is coarse - treat the band, not the figure.
    """
    words = segment(tag)
    if not words:
        return 0

    ranks = [rarity(word) for word in words]
    score = _BASE_LOG - _WORD_COST * (len(words) - 1)
    score -= sum(_RANK_COST[rank] for rank in ranks)
    if all(rank == 3 for rank in ranks):
        score -= _ALL_UNKNOWN_COST

    length = len(str(tag).lstrip("#"))
    if length > _LONG_TAG_CHARS:
        score -= _LONG_TAG_COST * ((length - _LONG_TAG_CHARS) / 8 + 1)

    return max(0, int(10 ** max(0.0, score)))


def band_of(count: int) -> str:
    """Which band a post count falls in."""
    if count < config.TAG_VOLUME_FLOOR:
        return "dead"
    if count <= config.TAG_VOLUME_NICHE_MAX:
        return "niche"
    if count <= config.TAG_VOLUME_MID_MAX:
        return "mid"
    if count <= config.TAG_VOLUME_BROAD_MAX:
        return "broad"
    return "saturated"


# ---------------------------------------------------------------------------
# The cache, and whatever provider was configured
# ---------------------------------------------------------------------------

def _read_json(path: Path, default: dict) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else dict(default)
    except (OSError, ValueError):
        return dict(default)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        with open(partial, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


class VolumeCache:
    """Post counts already known, keyed by lowercased tag.

    Hand-seeded entries and provider answers live in the same file. A
    hand-seeded one carries no `at`, so it never expires - if someone took the
    trouble to write a real figure down, a TTL should not throw it away.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or config.TAG_VOLUME_FILE)
        self._data = _read_json(self.path, {"version": 1, "tags": {}})
        if not isinstance(self._data.get("tags"), dict):
            self._data["tags"] = {}
        self._dirty = False

    def get(self, tag: str) -> tuple[int, str] | None:
        entry = self._data["tags"].get(str(tag).lower())
        if not isinstance(entry, dict):
            return None
        try:
            count = int(entry["count"])
        except (KeyError, TypeError, ValueError):
            return None
        if count < 0:
            return None
        source = str(entry.get("source") or "measured")
        if entry.get("at") and self._stale(str(entry["at"])):
            return None
        return count, source

    def put(self, tag: str, count: int, source: str) -> None:
        self._data["tags"][str(tag).lower()] = {
            "count": int(count),
            "source": source,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._dirty = True

    @staticmethod
    def _stale(stamp: str) -> bool:
        days = config.TAG_VOLUME_TTL_DAYS
        if days <= 0:
            return False
        try:
            when = datetime.fromisoformat(stamp)
        except ValueError:
            return True
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - when).days > days

    def save(self) -> list[str]:
        """Persist. A cache that cannot be written is a warning, not a failure."""
        if not self._dirty:
            return []
        try:
            _write_json(self.path, self._data)
        except OSError as exc:
            return [f"could not update the hashtag volume cache at "
                    f"{self.path}: {exc}"]
        self._dirty = False
        return []


def load_provider(path: str = ""):
    """Import the dotted path in TAG_VOLUME_PROVIDER, or return None.

    The callable is handed a list of tags and should return a mapping of tag
    to post count. Anything it leaves out falls through to the estimator.
    """
    target = (path or config.TAG_VOLUME_PROVIDER or "").strip()
    if not target:
        return None
    module_name, _, attr = target.rpartition(".")
    if not module_name or not attr:
        raise VolumeError(
            f"TAG_VOLUME_PROVIDER must be a dotted path like "
            f"mypackage.module.lookup - got {target!r}"
        )
    try:
        provider = getattr(import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise VolumeError(f"could not import TAG_VOLUME_PROVIDER {target!r}: "
                          f"{exc}") from exc
    if not callable(provider):
        raise VolumeError(f"TAG_VOLUME_PROVIDER {target!r} is not callable")
    return provider


def lookup(tags, *, cache: VolumeCache | None = None,
           provider=None) -> tuple[dict[str, Volume], list[str]]:
    """Resolve every tag to a Volume. Never raises for one bad answer.

    Order is cache, then provider, then the estimator. Provider answers are
    written back to the cache so the next run does not ask again.
    """
    warnings: list[str] = []
    wanted = [str(t).lstrip("#") for t in tags if str(t).strip()]
    if not wanted:
        return {}, warnings

    store = cache if cache is not None else VolumeCache()
    resolved: dict[str, Volume] = {}
    missing: list[str] = []

    for tag in wanted:
        hit = store.get(tag)
        if hit:
            count, source = hit
            resolved[tag] = Volume(tag, count, source, band_of(count))
        else:
            missing.append(tag)

    if missing and provider is None:
        try:
            provider = load_provider()
        except VolumeError as exc:
            warnings.append(str(exc))

    if missing and provider is not None:
        try:
            answers = provider(list(missing)) or {}
        except Exception as exc:                      # a provider is foreign
            warnings.append(f"hashtag volume provider failed, falling back to "
                            f"estimates: {exc}")
            answers = {}
        for tag in list(missing):
            raw = answers.get(tag, answers.get(tag.lower()))
            try:
                count = int(raw)
            except (TypeError, ValueError):
                continue
            if count < 0:
                continue
            store.put(tag, count, "provider")
            resolved[tag] = Volume(tag, count, "provider", band_of(count))
            missing.remove(tag)

    for tag in missing:
        count = estimate(tag)
        resolved[tag] = Volume(tag, count, "estimate", band_of(count))

    warnings += store.save()
    return resolved, warnings


# ---------------------------------------------------------------------------
# Standalone: see what it makes of a set of tags
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python pipeline/tagvolume.py <tag> [tag ...]",
              file=sys.stderr)
        return 2

    volumes, warnings = lookup(args)
    width = max(len(t.lstrip("#")) for t in args) + 2
    for tag in args:
        vol = volumes[tag.lstrip("#")]
        words = "+".join(segment(tag))
        print(f"#{vol.tag:<{width}} {vol.band:<10} ~{vol.count:>12,}  "
              f"{vol.source:<9} {words}")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
