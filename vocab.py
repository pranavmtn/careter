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
