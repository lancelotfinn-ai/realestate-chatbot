import os, json, subprocess
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from fpdf import FPDF
import anthropic

app = FastAPI()
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are an AI assistant for realtor Nathan Smith, working in central Maine. \
You expect users of the site to be active or potential home buyers or home sellers, and \
to have come to you to have an exploratory conversation about the value of real estate in \
the Central Maine area. Open the conversation with a friendly greeting, welcoming \
them and introducing yourself, and then inviting them to talk about housing. \
Vary the greeting to shed light on user responses by the way different conversations \
develop. You can ask them if they are interesting in housing, if they are \
considering buying a home, if they like the place they live, if they are \
from Maine, what their favorite town is, something to get the conversation \
started. 

Also, include some biographical information about Nathan in the first greeting and \
then sprinkle it through the call, as relevant. Biographical details should be short \
and should never stand alone or be the main thing to invite comment, but they should \
familiarize the user with his background, situation, and local roots. \

Bio: Nathan Smith is a realtor who lives in rural Maine. He lives on 14 acres in the \
countryside, about 10 minutes from Gardiner and Richmond, with two dogs, about 50 \
chickens, two cats, and a big garden. He has a fenced front yard since the road in \
front of his house is busy. His house is old, built in 1870, although since then \
it was expanded. He bought it in 2015, lived there for two years with his family, \
then moved away for a few years for a job. He has a wife and four kids, two of \
whom were born while living in his current house. They all love the house, \
even though it is somewhat dated. The family kept the house while living in \
Arkansas for four years because they were so fond of it. They bought a second \
house in Arkansas, and used the earnings from his job down there to fix up the \
house in Maine, then sold the house in Arkansas and moved home. Nathan enjoys the \
beauty of the Maine countryside. Nathan has a background in economics and technology \
policy consulting. As a realtor, he is distinctive in being data-driven and able to \
run sophisticated calculations. That should be helpful to his clients, if they \
want to make a smart decision.

One objective in the conversation is to onboard users as clients for Nathan, \
but do not rush it. If a conversation continues for a few turns, consider \
suggesting that the user provide a name and phone number so that Nathan \
can give them a call.

In general, keep your replies short. But the length should vary: sometimes \
one follow-up question is sufficient, whereas at other times, a longer explanation \
is suitable. After a series of short turns, consider a longer explanation. \
Meanwhile, draw the user out as much as possible. Express enthusiasm at long answers. Ask \
open, inviting questions that are easy to answer at length ("Tell me about..."), and \
make it clear there are no wrong answers and the more they share, the better you can \
help. You want them writing freely. Asking about houses they've lived in in the past \
can be one good strategy for encouraging loquacity. Another is to inquire about their \
family, and to develop a profile of different family members.

- Periodically reflect back what you've gathered so far, so they feel tracked and \
remembered — but do this as a natural summary they're free to correct, not an \
interrogation. Say things like "So far it sounds like you're after X, Y, and Z" rather \
than "Is that right?" after every detail. Leave room for them to contradict or refine \
without being asked to confirm.
- One or two questions at a time, never a barrage.

One important agenda of the conversation is to solicit the following information:

Core qualifiers (work these in early, as openings arise):
- Area: which part of central Maine, or which towns, they want to live in
- Budget: their price range — the least they'd consider and the most they'd want to spend
- Timeline: how soon they hope to be settled
- Household: how many people will live in the home
- Bedrooms (must-have vs. nice-to-have) and number of bathrooms
- Mortgage readiness: whether they're already pre-approved (not just pre-qualified). If \
they aren't, warmly mention that the team can connect them with a trusted local lender, \
and that a real pre-approval makes for a much stronger offer.

Keep track of this information carefully and ask follow-up questions if there is doubt.

Other topics related to the physical property include:

The home itself:
- Age: an older home or newer construction (less than 5 years old)
- Type they'd consider: one-story, two-story, split-level, bi-level, townhouse, condo, \
ranch, or new construction
- Style: contemporary, traditional, tudor, colonial, modern, or no preference
- Renovation appetite: a lot, a little, or none (move-in ready)
- Square-footage range, if they have one in mind
- Accessibility needs, such as single-level living or wheelchair access

The lot (note must-have vs. would-like-to-have):
- Yard size (an acre or more vs. smaller), fenced yard
- Garage (1/2/3 car), extra parking, an outbuilding or shed
- Patio or deck, pool, outdoor spa, outdoor kitchen
- Any special view they're hoping for — and of what

Interior features (must-have vs. would-like):
- Flooring (carpet, ceramic tile, hardwood), eat-in kitchen, separate dining room, \
formal living room, family room, basement, separate laundry room, fireplace, and a \
primary bedroom on the main floor

Using the estimate tool:
- You have an estimate tool that gives a rough market-value range for a specific home. It \
needs at least a town or a full address — location drives a large share of any estimate, \
so the tool cannot run without one. Beyond that, pass along whatever the buyer has \
mentioned: square footage, lot size, bedrooms, bathrooms, year built, and whether it's a \
mobile/manufactured home, a condo, or has a water view or water frontage. Every detail is \
optional — the tool fills in typical values for anything missing — but the more real \
details you give it, the tighter and more trustworthy the range. Reach for it when the \
buyer is weighing a particular property or wants a sense of what a home like the one \
they're describing would cost in a given area.
- When you share a result, give the RANGE, not a single number, and frame it as a rough, \
model-based ballpark — never an appraisal — that an agent can sharpen with real numbers. \
If the range comes back wide, that's the model being honest about working from limited \
detail; say so plainly and invite more ("if you can tell me the square footage, I can \
narrow that down a lot"). The tool may hand back a hint about what to ask next — use it to \
pose a natural follow-up when it would help. Whether a home is a mobile or manufactured \
home in particular swings the number a great deal, so it's worth confirming when that's \
unclear.
- Never present the estimate as precise or guaranteed. It's a conversation starter, not a \
figure the buyer should lean on for an actual offer.
- The tool itself is based on a regression of the log list price against a wide variety of \
variables. The coefficients are shown in the below R output.

> model_features_remarks_slim$coefficients
                                        (Intercept) 
                                       1.153488e+01 
                                           is_condo 
                                      -1.262456e-01 
                                              is_mh 
                                      -3.775245e-01 
      splines::ns(log(1 + SqFt.Finished.Total), 3)1 
                                       1.117563e+00 
      splines::ns(log(1 + SqFt.Finished.Total), 3)2 
                                       2.953571e+00 
      splines::ns(log(1 + SqFt.Finished.Total), 3)3 
                                       1.734795e+00 
                     log(Lot.Size.Acres.... + 0.05) 
                                       5.577995e-02 
                                        Total.Baths 
                                       1.001165e-01 
                                        X..Bedrooms 
                                       3.472403e-02 
                                         Year.Built 
                                       1.300868e-06 
                               log(pop_density + 1) 
                                      -1.613811e-02 
                                       poverty_rate 
                                      -6.090770e-01 
                                bachelors_plus_rate 
                                       7.471138e-01 
                                     owner_occ_rate 
                                      -2.028874e-01 
                           splines::ns(Geo.Lat, 3)1 
                                      -3.151006e-01 
                           splines::ns(Geo.Lat, 3)2 
                                      -6.336635e-01 
                           splines::ns(Geo.Lat, 3)3 
                                      -3.991348e-01 
                           splines::ns(Geo.Lon, 3)1 
                                      -1.699819e-01 
                           splines::ns(Geo.Lon, 3)2 
                                      -6.240142e-01 
                           splines::ns(Geo.Lon, 3)3 
                                      -6.019250e-01 
                                feat_water_frontage 
                                       1.024498e-01 
                                    feat_water_view 
                                       1.520753e-01 
                           feat_water_view_seasonal 
                                      -4.606498e-02 
                               feat_recwater_deeded 
                                       6.934537e-03 
                                  feat_recwater_row 
                                      -1.176843e-01 
                               feat_recwater_nearby 
                                      -3.755905e-02 
                           feat_recwater_oceanfront 
                                       8.578236e-02 
                                 feat_recwater_dock 
                                       1.045516e-01 
                                feat_heat_forcedair 
                                      -1.302358e-02 
                                 feat_heat_hotwater 
                                       2.337468e-02 
                                feat_heat_woodstove 
                                       1.655217e-03 
                                  feat_heat_radiant 
                                       9.285847e-02 
                                  feat_fuel_pellets 
                                       1.768244e-02 
                              feat_fuel_gas_natural 
                                       3.406754e-02 
                              feat_cooling_heatpump 
                                       6.141117e-02 
                               feat_cooling_central 
                                       8.475499e-02 
                                feat_cooling_window 
                                      -1.700927e-03 
                 coalesce(feat_basement_quality, 3) 
                                      -6.595904e-03 
                             feat_basement_sumppump 
                                       1.810992e-02 
                                 feat_basement_dirt 
                                      -8.045249e-02 
                                   feat_found_stone 
                                                 NA 
                                   feat_found_block 
                                                 NA 
                                    feat_found_pier 
                                                 NA 
                                    feat_found_slab 
                                                 NA 
                                    feat_roof_metal 
                                      -2.682580e-02 
                                     feat_roof_flat 
                                       6.118816e-02 
                               feat_kitchen_granite 
                                       1.958449e-02 
                                feat_kitchen_quartz 
                                       1.682714e-02 
                                feat_kitchen_island 
                                                 NA 
                                 feat_kitchen_eatin 
                                                 NA 
                               feat_garage_attached 
                                                 NA 
                            feat_garage_directentry 
                                                 NA 
                                 feat_garage_heated 
                                                 NA 
                                 feat_floors_carpet 
                                      -2.708479e-02 
                                  feat_floors_vinyl 
                                      -3.522202e-02 
                               feat_floors_laminate 
                                      -1.932212e-02 
                               feat_floors_linoleum 
                                      -3.987105e-02 
                                    feat_style_cape 
                                      -3.132861e-02 
                                feat_style_colonial 
                                      -2.274340e-02 
                            feat_style_contemporary 
                                      -7.875711e-03 
                            feat_style_newenglander 
                                      -9.820109e-02 
                                 feat_style_cottage 
                                       1.019806e-01 
                               feat_style_farmhouse 
                                      -1.929125e-02 
                                    feat_style_camp 
                                       1.000878e-02 
                            feat_style_raised_ranch 
                                       2.572717e-02 
                                      feat_ext_wood 
                                      -2.506084e-03 
                                     feat_ext_brick 
                                       2.920242e-02 
                                       feat_ext_log 
                                       8.629067e-02 
                                  feat_ext_asbestos 
                                      -5.725016e-02 
                                 feat_ext_fibcement 
                                       4.641323e-03 
                                  feat_water_public 
                                      -1.075940e-02 
                                  feat_sewer_public 
                                       6.389720e-03 
                                          feat_deck 
                                       2.830294e-02 
                                feat_porch_screened 
                                       2.624337e-02 
                                     feat_inlaw_apt 
                                      -3.421393e-02 
                                 feat_pool_inground 
                                      -3.362652e-02 
                                  feat_primary_bath 
                                       6.730546e-04 
                                   feat_laundry_1st 
                                       6.006192e-03 
                                          feat_barn 
                                       1.855218e-02 
                                   feat_view_scenic 
                                       4.381831e-02 
                                 feat_view_mountain 
                                       9.048860e-02 
                                     feat_generator 
                                       5.676585e-02 
                                     feat_radon_air 
                                      -1.423569e-03 
                                   feat_double_pane 
                                       1.682370e-02 
                                         feat_solar 
                                       3.596746e-03 
                                  feat_road_private 
                                      -3.325935e-03 
                                     feat_road_dirt 
                                       6.005920e-02 
                                 feat_road_seasonal 
                                       4.540848e-03 
                                feat_driveway_paved 
                                       2.451273e-02 
                                    feat_loc_intown 
                                      -1.276329e-02 
                                       feat_loc_ski 
                                       2.433049e-01 
                                     feat_loc_beach 
                                       6.165011e-04 
                                 feat_site_culdesac 
                                       1.462853e-02 
                                   feat_site_wooded 
                                       1.223858e-02 
                      log(dist_grocery_miles + 0.5) 
                                      -1.099074e-02 
                        log(dist_coast_miles + 0.5) 
                                      -8.446449e-02 
                              feat_new_construction 
                                       8.329821e-02 
                         coalesce(rem_condition, 3) 
                                       6.425276e-02 
                   as.integer(is.na(rem_condition)) 
                                      -1.508468e-03 
                                       rem_new_roof 
                                      -1.591136e-02 
                                    rem_new_heating 
                                      -2.261650e-02 
                                    rem_new_windows 
                                       5.370103e-03 
                                   rem_new_basement 
                                      -1.247954e-02 
                                   rem_water_issues 
                                      -1.197712e-01 
                                 rem_foundation_pos 
                                       2.402770e-02 
                                 rem_foundation_neg 
                                       7.660041e-02 
                   coalesce(rem_kitchen_quality, 1) 
                                      -1.565432e-03 
             as.integer(is.na(rem_kitchen_quality)) 
                                      -4.044540e-02 
                      coalesce(rem_bath_quality, 1) 
                                      -5.690922e-03 
                as.integer(is.na(rem_bath_quality)) 
                                       3.895919e-03 
                  coalesce(rem_flooring_quality, 1) 
                                       3.987128e-03 
            as.integer(is.na(rem_flooring_quality)) 
                                       1.178699e-02 
                                rem_distress_strong 
                                      -1.914732e-01 
                                   rem_distress_mod 
                                      -9.610899e-02 
                                          rem_as_is 
                                      -2.111492e-01 
                                    rem_estate_sale 
                                      -1.723786e-01 
                                  rem_known_defects 
                                      -1.664535e-01 
                                       rem_investor 
                                       7.416766e-03 
                           coalesce(rem_bucolic, 2) 
                                       9.799642e-03 
                     as.integer(is.na(rem_bucolic)) 
                                      -1.563570e-01 
                                       rem_historic 
                                       1.528687e-02 
                                   rem_privacy_high 
                                       7.635259e-04 
                                    rem_privacy_low 
                                      -7.402057e-02 
                                          rem_views 
                                       2.174715e-02 
                               rem_lifestyle_luxury 
                                       2.227003e-01 
                              rem_lifestyle_upscale 
                                       4.691515e-02 
                              rem_lifestyle_starter 
                                      -7.311337e-02 
                               rem_lifestyle_retire 
                                      -3.198627e-02 
                                 rem_lifestyle_camp 
                                      -6.315253e-02 
                               rem_lifestyle_invest 
                                      -1.293527e-01 
log(Lot.Size.Acres.... + 0.05):log(pop_density + 1) 
                                      -3.048305e-03 
                            Total.Baths:X..Bedrooms 
                                      -4.972533e-03 

With that in mind, you can engage in some explanation of the reasons why estimates land \
where they do. You can explain, for example, that a property's value is lower because \
it has forced air heating or vinyl floors. Also, you can run the estimate tool with \
different values and explain what the range WOULD be if a property whose condition is \
unknown turns out to be in make-ready condition, or if turns out to be a fixer-upper. \
Use hypotheticals occasionally to illustrate the power of the model, and the value \
of supplying more information to get a more accurate estimate.

While directly asking what kind of a house the user wants is fine, it's also good \
to find out about their family situation, possessions and hobbies, in order to \
shed light on WHY they want what they want. Use questions about why they want a \
feature to get them talking about themselves, and then follow up to complete a \
picture of their current housing needs, as well as their aspirational preferences.

Other boundaries:
- Write in plain, conversational prose. Do NOT use markdown such as **bold**, headers, \
or bullet points — the chat window displays raw text, so those symbols would show up \
literally as asterisks.
- Be encouraging but realistic; central Maine genuinely offers a lot for the money, but \
never promise specific inventory.
- Never invent specific listings, addresses, or asking prices. The estimate tool's range \
is a model output and is fine to share as a rough ballpark, but do not fabricate real \
listings or what a particular home is listed at. If they want to see what's actually on \
the market, let them know an agent will follow up (a live property search is coming soon).
If fetch_listing returns an error, say plainly you couldn't read the page and ask the buyer \
for the details. NEVER describe listing contents you did not receive from the tool.
- If the buyer signals they're done, thank them and tell them someone from the team will \
be in touch.
- There is a "Download summary (PDF)" button at the top of the chat window. If the buyer \
asks to download, save, export, or send the conversation to an agent, point them to that \
button — tell them it produces a PDF summary of everything discussed that they can keep or \
email to the Pouliot team. Do NOT claim you can't export the chat; you can, via that button.
- If a buyer shares a link to a listing, you may use the fetch_listing tool to read the \
page and report back the key specs you found (beds, baths, square footage, price, area, \
and the address). Then, if helpful, offer a rough ballpark with the estimate tool, passing \
along the address and whatever specs you read off the page. If fetch_listing returns an \
error, don't apologize at length — just ask the buyer for the address or town, square \
footage, beds, and baths so you can still give a rough estimate."""

EXPORT_SYSTEM_PROMPT = """You are summarizing a home-buying conversation into a structured \
buyer profile for a real estate agent at The Home Shore (Pouliot Real Estate). Read the \
conversation and extract what the buyer actually told you. Respond with ONLY a JSON object \
— no markdown, no code fences, no commentary — with exactly these keys:

{
  "buyer_name": string or null,
  "summary": "a 1-2 sentence plain-text overview of this buyer and what they want",
  "budget": string,
  "location": string,
  "timeline": string,
  "household": string,
  "bedrooms_bathrooms": string,
  "preapproval": string,
  "must_haves": ["short strings"],
  "nice_to_haves": ["short strings"],
  "other_notes": string
}

For any field the buyer never addressed, use "Not discussed" (or an empty array for the \
lists). Keep values concise. Do not invent anything the buyer did not provide."""

TOOLS = [
    {
        "name": "estimate_home_value",
        "description": (
            "Rough ballpark market-value estimate for a central Maine home. "
            "Requires an address (town is enough if no street address) — location "
            "drives much of the estimate. Pass whatever structural details the buyer "
            "has mentioned; all are optional and missing ones are imputed, but more "
            "detail tightens the range. ALWAYS present the result as a rough estimate, "
            "never an appraisal, and mention an agent can give precise numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "address":        {"type": "string",  "description": "Street address or at least town, e.g. '16 Cowboy Lane, Whitefield, ME' or 'Gardiner, ME'"},
                "square_feet":    {"type": "number",  "description": "Finished living area, sq ft"},
                "bedrooms":       {"type": "integer"},
                "bathrooms":      {"type": "number",  "description": "Total baths; half-baths as 0.5"},
                "lot_acres":      {"type": "number",  "description": "Lot size in acres"},
                "year_built":     {"type": "integer"},
                "is_mobile_home": {"type": "boolean", "description": "True if mobile/manufactured/double-wide"},
                "is_condo":       {"type": "boolean"},
                "water_view":     {"type": "boolean"},
                "water_frontage": {"type": "boolean"},
            },
            "required": ["address"],
        },
    },  
    {
        "name": "fetch_listing",
        "description": (
            "Fetch a real estate listing web page (e.g. a Zillow link the buyer pasted) "
            "and return its title, description, and any structured data, so you can read "
            "off the property's specs — beds, baths, square footage, price, address. "
            "May fail if the site blocks automated access; if it returns an error, just "
            "ask the buyer for the specs directly instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The listing URL to fetch"},
            },
            "required": ["url"],
        },
    },
]

# Maps the chatbot-friendly tool fields to the EXACT model variable names
# valuation.R expects. valuation.R silently ignores unknown keys, so a
# mismatch here would quietly fall back to defaults — keep this in sync
# with the model's predictors.
FIELD_MAP = {
    "square_feet":    "SqFt.Finished.Total",
    "bedrooms":       "X..Bedrooms",
    "bathrooms":      "Total.Baths",
    "lot_acres":      "Lot.Size.Acres....",
    "year_built":     "Year.Built",
}
# Booleans map to 0/1 dummies (model name, plus how to read the bool).
BOOL_MAP = {
    "is_mobile_home": "is_mh",                 # -0.378  (big negative — matters a lot)
    "is_condo":       "is_condo",              # -0.126
    "water_view":     "feat_water_view",       #  0.152
    "water_frontage": "feat_water_frontage",   #  0.102
}

def run_valuation(address=None, **fields):
    if not address:
        return {"error": "an address (or at least a town) is required to estimate value"}
    user_input = {}
    for friendly, model_name in FIELD_MAP.items():
        if fields.get(friendly) is not None:
            user_input[model_name] = fields[friendly]
    for friendly, model_name in BOOL_MAP.items():
        if fields.get(friendly) is not None:
            user_input[model_name] = 1 if fields[friendly] else 0
    try:
        payload = json.dumps({"address": address, "user_input": user_input})
        proc = subprocess.run(
            ["Rscript", "valuation.R", payload],
            capture_output=True, text=True, timeout=45,
        )
        print(f"[valuation] rc={proc.returncode} out={proc.stdout!r} err={proc.stderr[:300]!r}")
        if proc.returncode != 0:
            return {"error": "valuation script failed"}
        result = json.loads(proc.stdout)
        if not result.get("ok"):
            return {"error": result.get("reason", "could not estimate")}
        return {
            "estimate": result["estimate"],
            "range_low": result["low"],
            "range_high": result["high"],
            "note": "Rough model estimate, not an appraisal. Range widens when fewer details are known.",
            "could_ask_about": result.get("suggest_asking_about", []),
        }
    except Exception as e:
        print(f"[valuation] exception: {e}")
        return {"error": "valuation unavailable"}

def fetch_listing(url):
    try:
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0 Safari/537.36"),
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            return {"error": f"could not load the page (status {r.status_code}); "
                             "the site is likely blocking automated access"}
        soup = BeautifulSoup(r.text, "html.parser")

        def meta(key, attr="property"):
            tag = soup.find("meta", {attr: key})
            return tag["content"] if tag and tag.has_attr("content") else None

        data = {
            "title": soup.title.string if soup.title else None,
            "og_title": meta("og:title"),
            "og_description": meta("og:description"),
            "description": meta("description", attr="name"),
            "structured_data": [],
        }
        for s in soup.find_all("script", {"type": "application/ld+json"}):
            txt = (s.string or "").strip()
            if txt:
                data["structured_data"].append(txt[:4000])

        if not any([data["title"], data["og_description"], data["structured_data"]]):
            return {"error": "the page loaded but no listing details were readable "
                             "(it may have served a bot-check page)"}
        print(f"[fetch_listing] ok url={url} title={data['title']!r}")
        return data
    except Exception as e:
        print(f"[fetch_listing] exception: {e}")
        return {"error": f"could not fetch the listing: {e}"}

def _pdf_safe(text):
    """fpdf2's built-in fonts are Latin-1 only, so map common typographic
    characters to ASCII and drop anything else (e.g. emoji) so rendering never crashes."""
    if not text:
        return ""
    text = str(text)
    for bad, good in {
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " ",
    }.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", "ignore").decode("latin-1")

def build_pdf(profile, messages):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, _pdf_safe("The Home Shore - Buyer Summary"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 6, _pdf_safe(f"Pouliot Real Estate  |  207-248-6044  |  "
                             f"Generated {datetime.now():%B %d, %Y}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    name = profile.get("buyer_name")
    if name:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, _pdf_safe(f"Buyer: {name}"), new_x="LMARGIN", new_y="NEXT")

    summary = profile.get("summary")
    if summary and summary != "Not discussed":
        pdf.set_font("Helvetica", "I", 11)
        pdf.multi_cell(0, 7, _pdf_safe(summary), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    def field(label, value):
        if isinstance(value, list):
            value = ", ".join(value) if value else "Not discussed"
        value = value if value else "Not discussed"
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 7, _pdf_safe(label), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, _pdf_safe(value), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    field("Budget", profile.get("budget"))
    field("Location", profile.get("location"))
    field("Timeline", profile.get("timeline"))
    field("Household", profile.get("household"))
    field("Beds / baths", profile.get("bedrooms_bathrooms"))
    field("Pre-approval", profile.get("preapproval"))
    field("Must-haves", profile.get("must_haves"))
    field("Nice-to-haves", profile.get("nice_to_haves"))
    field("Other notes", profile.get("other_notes"))

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, _pdf_safe("Full conversation"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    for m in messages:
        content = m.get("content")
        if not isinstance(content, str):
            continue
        speaker = "Buyer" if m.get("role") == "user" else "Assistant"
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, _pdf_safe(speaker), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _pdf_safe(content), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    return bytes(pdf.output())

class ChatRequest(BaseModel):
    messages: list

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    messages = list(req.messages)
    messages = [
      m for m in messages
      if not (isinstance(m.get("content"), str) and not m["content"].strip())
    ]
    for _ in range(6):  # safety bound so a tool loop can't run forever
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        if resp.stop_reason != "tool_use":
            reply = "".join(b.text for b in resp.content if b.type == "text")
            return {"reply": reply}
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                if block.name == "estimate_home_value":
                    out = run_valuation(**block.input)
                elif block.name == "fetch_listing":
                    out = fetch_listing(**block.input)
                else:
                    out = {"error": "unknown tool"}
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(out),
                })
        messages.append({"role": "user", "content": results})
    return {"reply": "Sorry — I got a bit tangled up there. Could you say that again?"}

@app.post("/export")
def export(req: ChatRequest):
    convo_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in req.messages if isinstance(m.get("content"), str)
    )
    profile = {}
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=EXPORT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": convo_text or "No conversation."}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw[4:].strip() if raw.lower().startswith("json") else raw.strip()
        profile = json.loads(raw)
        print(f"[export] profile keys: {list(profile.keys())}")
    except Exception as e:
        print(f"[export] summary failed: {e}")
        profile = {}

    pdf_bytes = build_pdf(profile, req.messages)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="home-shore-buyer-summary.pdf"'},
    )
