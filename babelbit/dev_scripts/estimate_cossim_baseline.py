#!/usr/bin/env python3
"""
Compute a cosine-similarity baseline b ("unrelated ~= b") for a SentenceTransformer embedder.

Uses the user's provided embedder wrapper and cosine_similarity() implementation,
but computes b efficiently by embedding the unrelated sentence list once and then
sampling random pairs.

Environment variables:
  EMBEDDER_NAME   default: mixedbread-ai/mxbai-embed-large-v1
  EMBED_DIM       default: 64
  BASELINE_PAIRS  default: 20000
  BASELINE_SEED   default: 0
"""

import os
import random
import statistics
from typing import List, Tuple

from sentence_transformers import SentenceTransformer, util


# Constants with defaults - can be overridden by environment variables
EMBEDDER_NAME = os.getenv("EMBEDDER_NAME", "mixedbread-ai/mxbai-embed-large-v1")
EMBED_DIM = int(os.getenv("EMBED_DIM", "64"))

BASELINE_PAIRS = int(os.getenv("BASELINE_PAIRS", "20000"))
BASELINE_SEED = int(os.getenv("BASELINE_SEED", "0"))

# Lazy-loaded embedder (initialized on first use)
_embedder = None


def _get_embedder():
    """Get or initialize the sentence transformer model."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDER_NAME, truncate_dim=EMBED_DIM)
    return _embedder


def cosine_similarity(a: str, b: str) -> float:
    a = a or ""
    b = b or ""
    mx = max(len(a), len(b))
    if mx == 0:
        return 0.0
    embedder = _get_embedder()
    # Disable tqdm progress output from sentence-transformers in tight scoring loops
    ea = embedder.encode(a, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    eb = embedder.encode(b, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    cos = util.cos_sim(ea, eb).item()
    return cos


UNRELATED_SENTENCES_: List[str] = [
    "The kettle clicked off just as the rain started.",
    "A basalt cliff rises sharply above the black sand beach.",
    "I forgot my library card in the pocket of yesterday’s coat.",
    "The spreadsheet contains 17 columns and two empty rows.",
    "A violinist practiced scales behind a thin apartment wall.",
    "The recipe calls for two teaspoons of smoked paprika.",
    "A commuter train rolled past the graffiti-covered bridge.",
    "The telescope needed a new lens cap after the trip.",
    "He sketched a spiral staircase on the back of a receipt.",
    "My phone battery dropped from 42% to 11% in an hour.",
    "The museum guard yawned beside the marble statue.",
    "A fox crossed the lane and vanished into the hedgerow.",
    "The API returned HTTP 429 when the quota was exceeded.",
    "She rearranged the bookshelf by author surname.",
    "A ceramic mug cracked when it hit the tiled floor.",
    "The conference badge had a blue stripe and a QR code.",
    "An octopus can squeeze through surprisingly small gaps.",
    "The espresso machine hissed like a small steam engine.",
    "A faded postcard showed a lighthouse in winter.",
    "The garden hose tangled itself into a stubborn knot.",
    "Two cyclists argued about the fastest route to Cambridge.",
    "The violin case smelled faintly of pine resin.",
    "A new firmware update bricked the old router.",
    "He counted the steps: ninety-two from door to platform.",
    "The cat ignored the expensive toy and chased dust instead.",
    "A newspaper headline mentioned rising sea levels.",
    "The notebook margins were filled with tiny triangles.",
    "A thunderclap made the streetlights flicker once.",
    "She bought socks with little penguins stitched on them.",
    "The log file ended abruptly at 03:17:09 UTC.",
    "A robin hopped along the fence and pecked at moss.",
    "The pasta water boiled over and extinguished the flame.",
    "A chess clock ticked loudly in the quiet hall.",
    "He replaced the door hinge with stainless steel screws.",
    "A satellite image showed a hurricane’s eye clearly.",
    "The theatre program listed twelve scenes and one intermission.",
    "A single candle lit the table during the power cut.",
    "The elevator display alternated between 7 and 8.",
    "A barista drew a leaf in the foam with quick movements.",
    "The bicycle chain needed lubricant after the muddy ride.",
    "She misread the timetable and arrived a day early.",
    "A violin concerto drifted from an open window at dusk.",
    "The invoice total was £1,203.50 including VAT.",
    "A pine tree dropped needles onto the car bonnet.",
    "The bakery ran out of sourdough by 10 a.m.",
    "A metal detector beeped near a rusted bottle cap.",
    "The aquarium’s jellyfish pulsed under violet lights.",
    "He backed up the directory to an external drive.",
    "A page in the atlas was torn at the Arctic circle.",
    "The soup tasted better after resting overnight in the fridge.",
    "The lecture hall projector had a dead pixel in the center.",
    "A dog barked at its reflection in a glass door.",
    "The river current carried a branch downstream slowly.",
    "She learned the word 'sonder' and used it twice.",
    "A weather app predicted fog after midnight.",
    "The screwdriver tip was slightly magnetized.",
    "A stamp collection included one from 1936.",
    "The stadium lights turned the grass an unnatural green.",
    "He saved the draft as 'final_final_v3.docx'.",
    "A paper airplane landed in a cup of tea.",
    "The ferry schedule changed on Sundays in winter.",
    "She bought a map but still navigated by landmarks.",
    "A kettle of geese flew overhead in a V formation.",
    "The microwave beeped exactly three times.",
    "A marble rolled under the sofa and disappeared.",
    "The password manager suggested a 20-character string.",
    "A blank crossword grid waited on the kitchen table.",
    "He planted mint and immediately regretted it.",
    "The paint dried unevenly where the brush was overloaded.",
    "A paperback book absorbed the smell of sunscreen.",
    "The tram bell rang twice near the crowded stop.",
    "A small solder joint failed under vibration.",
    "She wore headphones but forgot to start the music.",
    "A compass needle jittered near the speaker magnet.",
    "The ice on the pond made a deep cracking sound.",
    "He compared two headphones using the same audio track.",
    "A train announcement echoed with a slight delay.",
    "The hard drive enclosure ran warm to the touch.",
    "A street vendor sold roasted chestnuts by the station.",
    "The calendar reminder triggered at 09:00 sharp.",
    "A moth circled the lamp and settled on the shade.",
]
UNRELATED_SENTENCES: List[str] =  [
    "Hi there, Ms. Lopez?",
    "Hi, I'm Dr. Stevens.",
    "Nice to meet you!",
    "Oh, hi Dr. Stevens, nice to meet you too.",
    "Welcome to the clinic.",
    "So I have your records here from Dr. Martin, but, uh, could you just start by telling me what's been going on?",
    "Yeah, sure.",
    "So, well, it's kind of complicated.",
    "For a few months now, maybe like four or five months, I've had these weird abdominal pains and just, uh, di- digestive stuff.",
    "Bloating, diarrhea, sometimes constipation.",
    "It's really unpredictable.",
    "Okay, got it.",
    "That sounds frustrating.",
    "Can you describe the abdominal pain a little more specifically?",
    "Yeah, so, it's usually here.",
    "Sometimes it's sharp, other times it's dull.",
    "It comes and goes.",
    "It's not consistent every day, but it is pretty frequent.",
    "Like, several times a week at least.",
    "Right, I see.",
    "And do you notice anything specific that seems to trigger it?",
    "Ccertain foods maybe, stress, or anything else?",
    "Mm, not always, but maybe stress.",
    "Food, I tried paying attention, but I couldn't really pin down anything specific.",
    "Sometimes dairy seems worse, but not always.",
    "Okay, interesting.",
    "Um, any weight changes recently?",
    "Uh, I've maybe lost a little unintentionally.",
    "Not a lot though, like 2kg in the last couple of months.",
    "Alright, good to know.",
    "Have you had fevers, night sweats, blood in the stool?",
    "Anything like that?",
    "No fevers or anything.",
    "Maybe occasionally I saw blood, but not a lot.",
    "Just, like, once or twice, a small amount.",
    "Okay, definitely worth noting.",
    "Um, I noticed Dr. Martin ran a few labs and imaging before referring you.",
    "Did she review those results with you yet?",
    "She said something about mild inflammation but didn't give me many details.",
    "Uh, it's partly why she sent me here.",
    "Alright, uh, let's look at those together.",
    "So your recent blood tests show mildly elevated CRP, uh that's an inflammatory marker, at about 12.",
    "It's- it's usually under 5.",
    "ESR is also mildly elevated at 30.",
    "Haemoglobin slightly low at 15, uh 15?",
    "11.5, suggesting maybe some mild anemia.",
    "Mm, okay.",
    "Is that…bad?",
    "Well, not necessarily bad, but it does suggest something inflammatory might be going on, maybe gastrointestinal-related.",
    "Uh, we'll have to explore more.",
    "Okay, good.",
    "Now, your stool tests were mostly negative.",
    "No parasites, infections, nothing obvious.",
    "So that's good news there.",
    "Yeah, Dr. Martin did say that was reassuring at least.",
    "Oh definitely reassuring.",
    "And your abdominal CT scan, let's see, here it says there's mild thickening in your, uh, terminal ileum.",
    "Uh, basically the end part of your small intestine.",
    "That can suggest inflammation, possibly Crohn's disease, but its- its not definitive from the imaging alone.",
    "Oh, okay.",
    "I wasn't quite clear on that.",
    "Yeah, imaging helps, but we usually need a bit more to confirm.",
    "Before we go further, let's do a quick physical exam.",
    "I'll talk through it if that's okay?",
    "Yeah, that's fine.",
    "Great.",
    "Alright, just lay back comfortably.",
    "General appearance: Ms. Lopez appears comfortable, slightly anxious but cooperative.",
    "Mild pallor noted.",
    "Abdomen: soft overall but mildly tender upon p- uh palpation in the right lower quadrant—right here, right?",
    "Yeah, that's exactly the spot.",
    "Okay, good.",
    "No masses felt, no obvious guarding.",
    "Bowel sounds normal.",
    "Quick check of eyes and mouth.",
    "Slight pallor in uh conjunctivae [laughter], consistent with mild anemia.",
    "Skin exam, no unusual rashes or lesions.",
    "That's good.",
    "Um, exam otherwise pretty normal.",
    "So, given your symptoms, labs, and imaging, I'm suspecting an inflammatory bowel disease, maybe Crohn's or another similar condition, but it's still not completely clear yet.",
    "Okay.",
    "So, how do we figure that out?",
    "Well, the best way to clarify this is usually a colonoscopy, basically a camera to directly visualise and biopsy the intestine.",
    "Have you had one before?",
    "No, I've never had one.",
    "But I've heard they're not very fun.",
    "Yeah, prep is the toughest part honestly.",
    "The procedure itself is quick, and you'll be asleep.",
    "But it really helps to give a definitive answer.",
    "Alright.",
    "Well, if you think it's important, I'll definitely do it.",
    "I really think it's our next best step.",
    "We'll biopsy any areas that look inflamed, which helps confirm or rule out specific conditions.",
    "Okay, that makes sense.",
    "Right.",
    "in the meantime, I want you to try a low-residue diet temporarily.",
    "Avoiding nuts, seeds, popcorn, raw veggies—things that might irritate your gut—just to see if symptoms calm down.",
    "Yeah, I've kinda been avoiding salad anyway because it made things worse.",
    "That's a good instinct, uh, listen to your body on that.",
    "Also, I'll prescribe you something mild to help with cramping, um, like dicycloverine, it- just as-needed until we get more answers.",
    "Uh, do you know of any medication allergies?",
    "No, no allergies.",
    "And that sounds fine.",
    "Great.",
    "And let's get some additional labs today: iron studies, vitamin B12 levels, and maybe celiac testing, just to rule everything else out.",
    "Okay, that sounds good.",
    "How's everything else going overall?",
    "You mentioned stress earlier, i- is that significant?",
    "Yeah, work's been stressful lately.",
    "New boss, deadlines, it probably doesn't help.",
    "It definitely can contribute.",
    "Stress doesn't cause inflammation, but it certainly can exacerbate symptoms.",
    "Managing stress might help it a bit.",
    "Any relaxation methods you like?",
    "Well, I used to do yoga, so maybe I should restart that.",
    "Yeah, that's a great idea.",
    "Uh, yoga, mindfulness, uh, they're definitely worth trying again.",
    "Yeah, I'll do that.",
    "Great.",
    "So, let's summarise our plan: First, we'll schedule a colonoscopy soon to clarify your diagnosis.",
    "Second, is the low-residue diet and, uh, dicycloverine as needed for symptoms.",
    "Third, additional labs: iron, B12, celiac screening.",
    "And fourth is stress reduction with yoga or mindfulness.",
    "And we'll review everything again after the colonoscopy.",
    "Okay, sounds very clear.",
    "Excellent.",
    "Any other questions or concerns?",
    "Urm, Nope, can't think of anything else right now.",
    "Alright, no worries.",
    "Uh, and call us anytime if you do think of anything.",
    "I'm glad you came in today.",
    "I know this can feel overwhelming, but we'll get you answers soon.",
    "Thanks, Dr. Stevens.",
    "I appreciate you being so thorough.",
    "Oh you're very welcome.",
    "Take care, Ms. Lopez.",
    "We'll talk soon!",
    "Thanks again, you too."
]


def _embed_all(sentences: List[str]):
    """
    Embed all sentences once, producing normalized embeddings suitable for fast cosine via dot product.
    Returns a tensor-like object; we keep it opaque and rely on sentence-transformers util.
    """
    embedder = _get_embedder()
    return embedder.encode(
        sentences,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def _sample_pairs(n_items: int, n_pairs: int, seed: int) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    pairs: List[Tuple[int, int]] = []
    for _ in range(n_pairs):
        i = rng.randrange(n_items)
        j = rng.randrange(n_items)
        if i == j:
            j = (j + 1) % n_items
        pairs.append((i, j))
    return pairs


def estimate_baseline_b(sentences: List[str], n_pairs: int, seed: int) -> dict:
    """
    Estimate b by:
      - embedding the sentence list once
      - sampling random pairs
      - averaging cosine similarities across those pairs

    Returns summary stats (mean=b, median, stdev, min, max, p05, p95).
    """
    if len(sentences) < 2:
        raise ValueError("Need at least 2 sentences to form pairs.")

    embs = _embed_all(sentences)
    pairs = _sample_pairs(len(sentences), n_pairs, seed)

    cos_vals: List[float] = []
    for i, j in pairs:
        # Since embeddings are L2-normalized, cosine is the dot product.
        # util.dot_score returns a 1x1 tensor-like; .item() gets the scalar.
        c = util.dot_score(embs[i], embs[j]).item()
        cos_vals.append(float(c))

    cos_vals_sorted = sorted(cos_vals)

    def pct(p: float) -> float:
        # p in [0,1]
        if not cos_vals_sorted:
            return float("nan")
        k = int(round(p * (len(cos_vals_sorted) - 1)))
        return float(cos_vals_sorted[k])

    out = {
        "n_sentences": len(sentences),
        "n_pairs": len(cos_vals),
        "seed": seed,
        "mean_b": float(statistics.fmean(cos_vals)),
        "median": float(statistics.median(cos_vals)),
        "stdev": float(statistics.pstdev(cos_vals)),
        "min": float(cos_vals_sorted[0]),
        "max": float(cos_vals_sorted[-1]),
        "p05": pct(0.05),
        "p95": pct(0.95),
    }
    return out


def main() -> int:
    print(f"Embedder: {EMBEDDER_NAME}")
    print(f"Truncate dim (EMBED_DIM): {EMBED_DIM}")
    print(f"Unrelated sentence count: {len(UNRELATED_SENTENCES)}")
    print(f"Sampling pairs: {BASELINE_PAIRS} (seed={BASELINE_SEED})")
    print()

    stats = estimate_baseline_b(UNRELATED_SENTENCES, BASELINE_PAIRS, BASELINE_SEED)

    print("Baseline estimate over random 'unrelated' pairs:")
    print(f"  b (mean cosine) : {stats['mean_b']:.6f}")
    print(f"  median         : {stats['median']:.6f}")
    print(f"  stdev          : {stats['stdev']:.6f}")
    print(f"  min / max      : {stats['min']:.6f} / {stats['max']:.6f}")
    print(f"  p05 / p95      : {stats['p05']:.6f} / {stats['p95']:.6f}")
    print()

    # Optional: show what your provided cosine_similarity() returns on a couple of arbitrary pairs
    # (this re-encodes per call, so keep it tiny).
    a, b = UNRELATED_SENTENCES[0], UNRELATED_SENTENCES[-1]
    c = cosine_similarity(a, b)
    print("Sanity check using cosine_similarity(a,b) on one pair:")
    print(f"  cos('{a[:32]}...', '{b[:32]}...') = {c:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
