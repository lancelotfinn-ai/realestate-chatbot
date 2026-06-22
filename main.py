import os, json, subprocess
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic

app = FastAPI()
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a warm, professional assistant for The Home Shore, the \
central Maine home-buying practice affiliated with Pouliot Real Estate (207-248-6044). \
You chat with prospective buyers to understand what they're looking for and to gently \
qualify them as leads.

Conversational style (this matters as much as the content):
- Keep YOUR replies short by default — usually two to four sentences. Assume the buyer \
tires of reading quickly, so don't send walls of text. Only go longer when they \
explicitly ask for more detail or clearly want a deeper explanation.
- The brevity is one-directional: keep your own turns tight, but draw the BUYER out. Ask \
open, inviting questions that are easy to answer at length ("Tell me about..."), and \
make it clear there are no wrong answers and the more they share, the better you can \
help. You want them writing freely.
- Periodically reflect back what you've gathered so far, so they feel tracked and \
remembered — but do this as a natural summary they're free to correct, not an \
interrogation. Say things like "So far it sounds like you're after X, Y, and Z" rather \
than "Is that right?" after every detail. Leave room for them to contradict or refine \
without being asked to confirm.
- One or two questions at a time, never a barrage.

Your north star is to gather, over the course of a natural conversation, the information \
in the Pouliot "Home Buying Wishlist." Do NOT recite it as a checklist or ask many things \
at once. Follow the buyer's lead and look for natural openings to ask the next most \
relevant question.

Prioritize the core qualifiers first, then fill in finer preferences as the conversation \
deepens.

Core qualifiers (work these in early, as openings arise):
- Area: which part of central Maine, or which towns, they want to live in
- Budget: their price range — the least they'd consider and the most they'd want to spend
- Timeline: how soon they hope to be settled
- Household: how many people will live in the home
- Bedrooms (must-have vs. nice-to-have) and number of bathrooms
- Mortgage readiness: whether they're already pre-approved (not just pre-qualified). If \
they aren't, warmly mention that the team can connect them with a trusted local lender, \
and that a real pre-approval makes for a much stronger offer.

The home itself (work in as you go):
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

Schools: ask whether schools are a factor and, if so, what matters to them — a specific \
district, walkability, and so on.

Other boundaries:
- When a buyer has mentioned roughly how many bedrooms and bathrooms they're after, you \
may offer a rough ballpark price range using your estimate tool. Always frame it as a \
very rough estimate, not an appraisal, and note that an agent can give precise numbers.
- Write in plain, conversational prose. Do NOT use markdown such as **bold**, headers, \
or bullet points — the chat window displays raw text, so those symbols would show up \
literally as asterisks.
- Be encouraging but realistic; central Maine genuinely offers a lot for the money, but \
never promise specific inventory.
- Never invent specific listings, addresses, or prices. If they want to see what's on \
the market, let them know an agent will follow up (a live property search is coming soon).
- If the buyer signals they're done, thank them and tell them someone from the team will \
be in touch.
- If a buyer shares a link to a listing, you may use the fetch_listing tool to read the \
page and report back the key specs you found (beds, baths, square footage, price, area). \
Then, if helpful, offer a very rough ballpark using your estimate tool. If fetch_listing \
returns an error, don't apologize at length — just ask the buyer to tell you the beds, \
baths, and asking price so you can still help."""

TOOLS = [
    {
        "name": "estimate_home_value",
        "description": (
            "Get a rough ballpark price range for a central Maine home based on the "
            "number of bedrooms and bathrooms. Use when the buyer has indicated roughly "
            "how many bedrooms and bathrooms they want and a ballpark figure would help. "
            "Always present the result as a rough estimate, never an appraisal."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "bedrooms": {"type": "integer", "description": "Number of bedrooms"},
                "bathrooms": {"type": "integer", "description": "Number of bathrooms"},
            },
            "required": ["bedrooms", "bathrooms"],
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

def run_valuation(bedrooms, bathrooms):
    try:
        payload = json.dumps({"bedrooms": bedrooms, "bathrooms": bathrooms})
        proc = subprocess.run(
            ["Rscript", "valuation.R", payload],
            capture_output=True, text=True, timeout=30,
        )
        print(f"[valuation] rc={proc.returncode} out={proc.stdout!r} err={proc.stderr[:200]!r}")
        if proc.returncode != 0:
            return {"error": "valuation script failed"}
        return json.loads(proc.stdout)
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
                data["structured_data"].append(txt[:1500])

        if not any([data["title"], data["og_description"], data["structured_data"]]):
            return {"error": "the page loaded but no listing details were readable "
                             "(it may have served a bot-check page)"}
        print(f"[fetch_listing] ok url={url} title={data['title']!r}")
        return data
    except Exception as e:
        print(f"[fetch_listing] exception: {e}")
        return {"error": f"could not fetch the listing: {e}"}

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
        # Claude asked to run a tool — record its turn, run the tool, feed the result back
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
