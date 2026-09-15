# TODO za PR: environment/setup fixevi za Pasado repo

Beleške napravljene 2026-07-06 nakon što sam podigao radni environment za `Section_5_2/climate.sh`
na Windows 11 / Python 3.13.3. Ovo NIJE PR, samo checklist za admina koji priprema PR.

---

## 1. Dodati `requirements.txt` (repo trenutno nema nijedan)

Repo (`uiuc-arc/Pasado`) nema ni `requirements.txt` ni `pyproject.toml` — samo markdown
tabelu verzija u README-u ("Requirements" sekcija). To znači da se ne može uraditi
`pip install -r requirements.txt`, korisnik mora ručno da instalira svaki paket.

**Verzije koje sam ja testirao i potvrdio da rade** (Python 3.13.3, 2026-07-06, sve najnovije
dostupne na PyPI u tom trenutku, bez pinovanja na stare README verzije):

```
numpy==2.5.1
scipy==1.18.0
scikit-learn==1.9.0
matplotlib==3.11.0
seaborn==0.13.2
tabulate==0.10.0
tqdm==4.68.3
torch==2.12.1
torchvision==0.27.1
torchaudio==2.11.0
affapy==0.1
jupyter==1.1.1
ipython==9.15.0
nbconvert==7.17.1
nbformat==5.10.4
```

**Odluka za admina:** pinovati tačne verzije (bolja reproducibilnost) ili ostaviti `>=` raspon
kao u README-u (bolja buduća kompatibilnost)? Predlažem donji prag (`>=`) da se ne lomi opet za
2-3 godine, uz napomenu da su gornje verzije poslednje testirane.

---

## 2. Ažurirati README "Requirements" tabelu (linije ~28-46)

Trenutna tabela je iz 2023 (Python 3.10.6, numpy 1.25.1, sklearn 1.3.0, torch 2.0.1...).
Testirao sam i **potvrdio da `climate.sh` radi bez grešaka** i sa mnogo novijim verzijama
(vidi tabelu iznad, Python 3.13.3 uključen). Trebalo bi dodati red/napomenu da je testirano i
na ovim novijim verzijama, ne samo "3.9 ili novije" bez konkretne potvrde.

**Napomena/rizik koji treba dodatno proveriti pre nego što se README ažurira:**
`chemical.sh` (za razliku od `climate.sh`) učitava `saved.pkl` — istreniran `MLPRegressor`
pickle-ovan sa scikit-learn 1.3.0. Već sa sklearn 1.6.1 se u starom `run_all.log`-u javljalo
`InconsistentVersionWarning` pri unpickle-ovanju. Sa sklearn 1.9.0 (koje sam ja instalirao)
razlika je još veća — **nisam proveravao da li `chemical.sh` i dalje daje identične brojeve**,
samo da li se import-uje/pokreće. Pre nego što se README zvanično ažurira da podržava
sklearn 1.9.x, neko treba da pokrene `chemical.sh` i uporedi izlazne brojeve sa referentnim
rezultatima iz rada/starog loga.

---

## 3. Pravi bug: PYTHONPATH nije podešen u `climate.sh` / `chemical.sh`

`climate_ode_v2.py` i `chemical_example.py` (u `Section_5_2/`) rade:
```python
from dual_intervals import *
from runge_kutta import *
```
Ovi moduli fizički postoje u `../forward_mode_non_tensorized_src/`, koji **nije** na
`PYTHONPATH` kad se skripta pokrene iz `Section_5_2/` sa `python3 climate_ode_script.py`.
Bez ručne intervencije (PYTHONPATH env var, IDE "sources root", ili `.pth` fajl u
site-packages — ja sam koristio ovo poslednje kao lokalni workaround) import puca sa
`ModuleNotFoundError: No module named 'dual_intervals'`.

**Predlog fix-a** — dodati na vrh `climate.sh` i `chemical.sh` (odmah posle `#!/bin/sh`):
```sh
export PYTHONPATH="$(cd "$(dirname "$0")/../forward_mode_non_tensorized_src" && pwd):$PYTHONPATH"
```
Portabilno (Linux/Mac/Git Bash), ne zahteva izmenu `.py` fajlova, i rešava problem za
svakog ko klonira repo iznova.

---

## 4. Windows kompatibilnost (opciono, van scope-a osim ako se eksplicitno želi)

`climate.sh`/`chemical.sh` zovu `python3`, što na Windows venv-ovima ne postoji (venv pravi
samo `python.exe`, ne `python3.exe`). Ja sam lokalno napravio kopiju `python3.exe` kao
workaround — ovo NIJE deo repo-a i ne mora ići u PR, osim ako žele da eksplicitno
dokumentuju Windows setup u README-u (npr. "on Windows, run `copy python.exe python3.exe`
in your venv's Scripts folder, or use WSL/Git Bash").

---

## Referenca: šta sam ja lokalno uradio (van repo-a, ništa nije committed)

- Napravio `.venv/` u `Pasado/` i instalirao gorenavedene pakete (nepinovano, najnovije verzije).
- Dodao `.pth` fajl u `.venv/Lib/site-packages/` koji upućuje na `forward_mode_non_tensorized_src`
  (lokalni workaround za tačku 3, nije deo repo-a).
- Dodao `python3.exe` (kopija `python.exe`) u `.venv/Scripts/` (lokalni Windows workaround za tačku 4).
- `climate.sh` sam pokrenuo end-to-end i potvrdio da radi (generisao `img/` i `data/` fajlove).
- Git status Pasado repo-a nije menjan — sve gorenavedeno je van git trackinga (`.venv/` je već
  u `.gitignore`).
