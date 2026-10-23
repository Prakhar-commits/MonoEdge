"""Generate the sample documents used by the evaluation (deterministic output).

Run from the repo root:  python eval/make_sample_docs.py

Every fact lives on exactly one known page, so each question has one correct (file, page).
The handbook is worded so that likely questions use DIFFERENT words from the text
(paraphrase tests). The FAQ is full of exact identifiers (keyword-search tests).
Note: the PDF uses "Rs" instead of the rupee sign because the built-in PDF fonts cannot draw it.
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT_DIR = Path(__file__).parent / "docs"

HANDBOOK_PAGES = [
    (
        "Welcome to Acme",
        [
            "Acme Labs builds workflow software for logistics companies. We have offices in "
            "Bengaluru, Pune and Singapore, and about 180 people across product, engineering, "
            "sales and support.",
            "This handbook explains how we work together. It covers leave, expenses, security, "
            "on-call duty, equipment and remote work. If something is unclear, ask your manager "
            "or write to people@acme.example.",
            "Our working day runs from 9:30 to 18:00 local time, Monday to Friday. Public "
            "holidays follow the calendar of the office where you are based.",
        ],
    ),
    (
        "Leave policy",
        [
            "Full-time employees receive 24 days of paid annual leave in each calendar year. "
            "Leave builds up month by month from the joining date, so new joiners receive a "
            "proportional amount in their first year.",
            "Up to 5 unused days can be carried over into the next calendar year. Anything above "
            "that expires on 31 March. Leave requests should reach your manager at least two "
            "weeks in advance for absences longer than five days.",
            "Sick leave is separate from annual leave: 10 days per year. A doctor's certificate "
            "is required once an illness lasts more than 3 consecutive days.",
        ],
    ),
    (
        "Expenses and reimbursement",
        [
            "Staff can claim back money spent on business trips, client meals and approved "
            "training. Keep every receipt and upload it to the expense portal within 30 days "
            "of the spend.",
            "Any single expense above Rs 5,000 must be approved by the finance team before the "
            "money is spent. Smaller expenses only need approval from your line manager.",
            "Approved claims are paid into your salary account within 10 working days. Claims "
            "without a receipt are rejected, except taxi rides under Rs 200.",
        ],
    ),
    (
        "Information security",
        [
            "Passwords must be rotated every 90 days, and the new password may not repeat any of "
            "your last five. Multi-factor authentication is mandatory for email, the code "
            "repository and the production consoles.",
            "All laptops use full-disk encryption. Screens lock automatically after 5 minutes "
            "of inactivity. Never store customer data on personal devices or in personal cloud "
            "storage.",
            "Report a lost laptop or a suspicious email to security@acme.example immediately; "
            "do not wait until the next working day.",
        ],
    ),
    (
        "On-call duty",
        [
            "Engineers on the platform team take turns being on call. A rotation lasts one week "
            "and the hand-over happens on Monday at 10:00 in the engineering channel.",
            "If a page is not acknowledged within 30 minutes, it escalates automatically to the "
            "engineering manager. The person on call should stay within reach of a laptop and a "
            "stable internet connection for the whole week.",
            "Each week of on-call earns one additional day off, to be taken within the following "
            "two months.",
        ],
    ),
    (
        "Hardware and equipment",
        [
            "Laptops are replaced every 36 months. To request a new machine or a replacement, "
            "open a ticket with the IT service desk; delivery normally takes 5 working days.",
            "Accessories such as keyboards, mice and headsets costing less than Rs 3,000 need "
            "only your manager's approval. Monitors are provided for office desks only.",
            "When you leave Acme, return all equipment within 7 days of your last working day.",
        ],
    ),
    (
        "Remote work",
        [
            "Most roles can work remotely up to 3 days per week. Please be reachable during core "
            "hours, 11:00 to 16:00 local time, regardless of where you work.",
            "Working from another country for more than 14 days needs written approval from "
            "your manager and the people team, because of tax and legal rules.",
            "A monthly stipend of Rs 1,500 helps cover internet costs for employees who work "
            "remotely at least two days a week.",
        ],
    ),
]

# A second PDF made of DECOYS: its pages share vocabulary with the handbook (approval, alert,
# credentials, leave, error codes) but state different facts, so retrieval has to choose well.
PLAYBOOK_PAGES = [
    (
        "Release process",
        [
            "Production releases go out every Tuesday between 11:00 and 13:00. Any deployment "
            "outside that window must be approved by the release manager before it starts.",
            "Every release needs a rollback plan written in the pull request. Releases are "
            "announced in the engineering channel at least one hour in advance.",
            "Hotfixes for customer-facing bugs may skip the weekly window but still need a "
            "second reviewer.",
        ],
    ),
    (
        "Incident severity levels",
        [
            "Alerts are classified by severity. A severity-1 alert means customers cannot use "
            "the product at all; an incident commander is named and a status page update is "
            "published every 30 minutes until the issue is resolved.",
            "A severity-2 alert means a major feature is degraded. Status page updates are "
            "published every 2 hours. Severity-3 alerts are handled during working hours.",
            "After every severity-1 or severity-2 incident the team writes a blameless review "
            "within 5 working days.",
        ],
    ),
    (
        "Code review",
        [
            "Every change needs one approval from a teammate before it is merged. Changes to "
            "infrastructure code, billing or authentication need two approvals, one of them "
            "from a code owner.",
            "Reviewers aim to respond within one working day. If a review has had no response "
            "for two days, ask in the team channel.",
            "Large changes over 800 lines should be split into smaller pull requests.",
        ],
    ),
    (
        "Cloud cost control",
        [
            "Teams are responsible for their cloud bill. Any new cloud service that will cost "
            "more than 500 US dollars per month must be approved by the engineering lead before "
            "it is created.",
            "Cost reports are sent on the first Monday of every month. Unused test environments "
            "are shut down automatically after 14 days.",
            "Reserved capacity purchases are decided quarterly by the finance and platform "
            "teams together.",
        ],
    ),
    (
        "Service credentials and secrets",
        [
            "API keys and service credentials live in the secrets manager and are rotated every "
            "180 days by the platform team. Never commit credentials to the repository.",
            "Each service has its own credentials; sharing a key between two services is not "
            "allowed. Revoke a credential immediately if it appears in logs or a screenshot.",
            "Access to production databases is granted for 8 hours at a time through the "
            "access request tool.",
        ],
    ),
    (
        "Learning and conferences",
        [
            "Engineers receive 5 paid days per year for training, courses and conferences. "
            "These days are in addition to normal annual leave and do not carry over.",
            "The learning budget is Rs 40,000 per person per year. Choose a course or event, "
            "write two lines on why it helps your work and share it with your manager.",
            "After a conference, give a short talk to the team within a month.",
        ],
    ),
    (
        "Engineering onboarding",
        [
            "New engineers get a buddy for the first 60 days. In week one you set up your "
            "laptop, get access to the code repository and ship a small change to production.",
            "During the first month you shadow a teammate during their on-call week before "
            "joining the rotation yourself in month three.",
            "Your manager holds a check-in with you every week for the first three months.",
        ],
    ),
    (
        "Internal error codes",
        [
            "Internal services report problems with short codes. I-4012 means the job queue is "
            "full and new jobs are being delayed. I-4013 means a worker lost its database "
            "connection.",
            "I-5020 means a scheduled export failed and will be retried automatically. I-5021 "
            "means the retry limit was reached and a person must restart the export.",
            "These codes appear in internal logs only. Customers see the public error codes "
            "documented in the product FAQ.",
        ],
    ),
]

FAQ_TEXT = """Acme Flow Product FAQ

Plans and API limits

Acme Flow is sold in three plans: Starter, Growth and Enterprise. The Starter plan allows 100 API requests per minute and 5 projects. The Growth plan allows 1,000 API requests per minute and 50 projects. The Enterprise plan has custom limits that are agreed in the contract.

Error codes

E-4012: the API token has expired. Generate a new token from Settings > API Keys and update your integration.

E-5001: rate limit exceeded. The request was rejected because the plan's requests-per-minute limit was reached. Retry after the number of seconds given in the Retry-After header.

E-6030: webhook signature mismatch. The X-Signature header did not match the HMAC-SHA256 of the request body. Check that the signing secret is correct and that the body was not modified before verification.

E-7100: file too large. Uploads are limited to 25 MB on the Starter plan and 100 MB on the Growth plan.

Support

Starter customers receive email support with a response within 2 business days. Growth customers receive a response within 8 business hours. Enterprise customers have a named support engineer and a 1 hour response target for severity-1 incidents.
"""


def build_pdf(path: Path, title: str, pages: list) -> None:
    styles = getSampleStyleSheet()
    # invariant=1 removes the creation timestamp so regenerating gives an identical file.
    doc = SimpleDocTemplate(str(path), pagesize=A4, invariant=1, title=title)
    story = []
    for index, (heading, paragraphs) in enumerate(pages):
        story.append(Paragraph(heading, styles["Heading1"]))
        for text in paragraphs:
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 10))
        if index < len(pages) - 1:
            story.append(PageBreak())
    doc.build(story)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    build_pdf(OUT_DIR / "acme_handbook.pdf", "Acme Handbook", HANDBOOK_PAGES)
    build_pdf(OUT_DIR / "acme_engineering_playbook.pdf", "Acme Engineering Playbook", PLAYBOOK_PAGES)
    (OUT_DIR / "acme_product_faq.txt").write_text(FAQ_TEXT, encoding="utf-8")
    print(f"Wrote sample documents to {OUT_DIR}")


if __name__ == "__main__":
    main()
