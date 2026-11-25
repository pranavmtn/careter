"""Everyday work, tech and office words. They're ranked higher when you start typing them (so "po"
offers "portal", "de" offers "deploy"), are always known words for auto-correct, and get added to the
word lists if missing (jira, hotfix, llm...). Edit freely: one word per space."""

BOOST = frozenset("""
bug bugs portal portals deploy deployed deployment server servers database databases api apis endpoint
endpoints frontend backend repo repository branch branches merge merged commit commits release releases
build builds pipeline staging production testing tester debug debugging issue issues ticket tickets
sprint standup jira github gitlab docker kubernetes cloud aws azure login logout signup password
username dashboard admin feature features requirement requirements module modules workflow workflows
integration automation script scripts config configuration update updates upgrade install installation
dependency dependencies version versions patch hotfix crash crashed error errors exception timeout
latency performance optimize refactor review reviewed approval approve approved feedback documentation
docs link links url website webpage app apps mobile android ios browser chrome extension plugin
notification notifications email emails attachment attachments invoice invoices payment payments
checkout cart order orders customer customers client clients vendor vendors quotation proposal estimate
budget deadline deadlines schedule scheduled meeting meetings agenda presentation slides report reports
spreadsheet excel document documents pdf folder drive shared access permission permissions team
teammate colleague colleagues manager project projects task tasks milestone roadmap priority urgent
asap eod fyi regards thanks please kindly noted confirm confirmation available availability tomorrow
today yesterday weekend morning evening onboarding training interview resume salary leave holiday
office remote laptop desktop keyboard screen screenshot upload uploaded download downloaded sync backup
restore delete deleted remove removed rename edit edited save saved submit submitted cancel cancelled
retry refresh reload reset verify verification otp analytics metrics kpi data dataset model models
prompt prompts chatbot agent agents llm ai gpt token tokens embedding embeddings vector inference
accuracy prediction predictions suggestion suggestions autocomplete frontend ui ux design designer
wireframe mockup prototype figma component components responsive layout template templates domain
hosting ssl dns cache caching queue webhook webhooks cron logs logging monitoring alert alerts
incident outage downtime uptime migration migrate schema query queries json csv xml yaml python
javascript typescript react node flask django java kotlin swift sql postgres mysql mongodb redis
firebase supabase vercel netlify nginx linux windows macos terminal command commands localhost
codebase pull request requests approve assignee reporter blocker blocked resolved reopen reopened
duplicate testcase testcases regression qa uat sanity smoke scenario scenarios checklist
stakeholder stakeholders deliverable deliverables scope estimate estimates invoice gst tax quote
whatsapp telegram slack teams zoom gmail outlook calendar invite invitation reschedule postpone
followup reminder discussed discussion clarify clarification regarding attached forwarding
""".split())


# Manglish (Malayalam typed in English letters). Words are offered when you start typing them and never
# auto-corrected; phrases teach the next word ("pinne " offers "kaanam"). Settings → Manglish edits this
# list (it's saved in settings.json); this is the starting list. Separate entries with commas.
MANGLISH_DEFAULT = """
Sughamano, Sugham, Enthokke undu, Enthaa, Evide, Eppo, Aara, Enthina, Engane, Sheri, Aa, Aanu,
Alla, Illa, Undu, Nanni, Pinne kaanam, Poyitt varaam, Machane, Machaa, Aliya, Chetta, Chechi,
Mone, Mole, Pwoli, Kidu, Polichu, Adipoli, Scene illa, Scene aanu, Vibe aanu, Thallu,
Thalli marikkal, Chali, Mass, Kalippu, Kalli, Pani kitti, Seen aakki, Ayyo, Ente ponno, Mathi,
Vende, Venam, Pettennu, Pathukke, Njan varaam, Njan busy aanu, Call cheyyam, Message ayakk,
Ready aayo, Ithu set aanu, Tension adikkanda, Free aano, Food kazhicho, Late aakum,
Pinne paranja mathi, Onnum manasilayilla, Enikk ariyilla, Kuzhappam illa, Vere level,
Download cheyyu, Share cheyyam, Delete cheythu, Kollaam, Ponnu, Chunk, Muthe, Kutta, Eda, Edi,
Aashaan, Uncle, Aunty, Kashtam, Bore aanu, Ozhivu, Pokkam, Vaa, Po, Nokk, Para, Kelk, Irikk,
Kazhikk, Kudikk, Urangu, Ezhunnelkk, Sathyam, Vishwasikk, Ishtam, Pedi, Sankadam, Santhosham,
Poli sanam, Thakarthu, Kalakki, Pottan, Mandan, Kozhi, Kaasu, Panam, Vandi, Veedu, Chaya, Choru,
Vellam, Mazha, Choodu, Thanupp
"""


def manglish_index(text):
    """'Pinne kaanam, Sheri' -> (all words, {word: [next words]}, phrases in order)."""
    import re
    phrases = [" ".join(p.lower().split()) for p in re.split(r"[,;\n]+", text or "") if p.strip()]
    words, pairs = [], {}
    for p in phrases:
        ws = [w for w in re.findall(r"[a-z']+", p)]
        for w in ws:
            if w not in words:
                words.append(w)
        for a, b in zip(ws, ws[1:]):
            nxt = pairs.setdefault(a, [])
            if b not in nxt:
                nxt.append(b)
    return words, pairs, phrases
