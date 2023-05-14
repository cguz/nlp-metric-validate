from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from nltk.translate.bleu_score import corpus_bleu
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/calculate_metric", response_class=HTMLResponse)
async def calculate_metric(request: Request, metric: str = Form(...), candidate: str = Form(...), reference: str = Form(...)):
    # Calculate the BLEU score using NLTK
    # Assumes that candidate and reference sentences are already tokenized
    bleu_score = corpus_bleu([[reference]], [candidate])

    return templates.TemplateResponse("result.html", {"request": request, "metric": metric, "candidate": candidate, "reference": reference, "bleu_score": bleu_score})