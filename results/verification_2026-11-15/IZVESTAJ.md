# Kako smo proverili rezultate i zašto baš tako

Ovaj izveštaj objašnjava, običnim jezikom, kako je provereno ono što tvrdim o ubrzanju Pasada i o ispravnosti njegovih granica. Za svaku proveru piše na koje pitanje odgovara, zašto je izabrana baš ona, kako je urađena i šta je pokazala. Tehnički detalji, brojevi i logovi su u [README.md](README.md) u istom folderu; [IZVESTAJ.md](IZVESTAJ.md) je ovaj tekst.

## 1. Šta smo hteli da znamo

Rad tvrdi tri stvari. Svaka traži drugu vrstu provere.

1. **Brojevi o brzini su tačni.** Ako kažem „korak X je ubrzao Y puta“, to mora da važi i kad neko drugi preračuna, i kad se merenje ponovi.
2. **Granice su i dalje ispravne (sound).** Pasado daje gornju granicu za to koliko se izlaz mreže može promeniti pod izmaglicom. Granica sme da bude preširoka, ali nikad ne sme da bude ispod stvarne promene. Svaka optimizacija mora to da sačuva.
3. **Poređenja su poštena.** Ubrzanje mereno između dve različite mašine, ili u odnosu na lošu polaznu tačku, može da izgleda mnogo veće nego što jeste.

Opšte pravilo kojim sam se vodio ima tri dela:

- **Nijednom izvoru ne verujem dvaput.** Isti zaključak proveravam drugim putem, drugim kodom ili drugom mašinom.
- **Test vredi samo ako može da padne.** Zato sam namerno kvario kod i gledao da li testovi to primete.
- **Svaki čudan broj tražim do uzroka.** Ne ostavljam ga kao „verovatno šum“.

## 2. Brzina: da li su brojevi tačni

### 2.1 Svaki broj preračunat iz sirovih merenja

**Šta:** Skripta za merenje snima svaki pojedinačni prolaz (3.070 prolaza u glavnom runu), pa iz njih pravi tabelu. Napisao sam posebnu, nezavisnu skriptu koja iz istih sirovih merenja ponovo računa svaki broj iz prezentacije: vremena, ubrzanja, raspone i procente.

**Zašto:** Tabelu pravi kod, a kod može da ima grešku. Ako ista greška postoji i u merenju i u proveri, nećemo je videti. Zato provera ne sme da koristi kod koji proverava.

**Rezultat:** Svi brojevi se poklapaju. Ova provera je ranije našla stvarnu grešku u izveštaju. Za FFNNBig sa 64 splita, poslednji korak je bio usrednjen preko 6 parova (slika, ε) umesto preko 2 zajednička, pa je granica u tabeli izgledala drugačija nego što jeste. Ispravljeno je pre nego što je bilo šta ušlo u prezentaciju.

### 2.2 Da li svaki korak stvarno izvršava ono što tvrdi

**Šta:** Lestvica ima 9 koraka: original, vektorizacija, batched gelsd, realna kubna formula, GPU, batched splitovi, zatvorena forma, kompajlirani fit i kompajlirani sigmoid. Koraci se uključuju prekidačima u kodu. Za svaki korak sam prebrojao koje funkcije se stvarno pozivaju i proverio dve stvari: da se pozivaju one koje treba, i da se ne pozivaju one koje ne treba.

**Zašto:** Prekidač koji se ne uključi ne baca grešku. Program radi dalje, samo meri pogrešnu stvar. Najopasnije je kompajliranje: kad ne uspe, kod tiho prelazi na nekompajliranu verziju. Merili bismo „kompajlirani“ korak koji to nije.

**Rezultat:** Svih 9 koraka poziva tačno ono što tvrde. Kompajlirani fit zaista radi kompajliran, a kompajlirani sigmoid izvršava jedan kompajlirani graf.

### 2.3 Da li štoperica čeka GPU

**Šta:** Za svaki GPU korak uporedio sam koliko je GPU stvarno bio zauzet (iz profilera) sa vremenom koje je izmerila štoperica.

**Zašto:** GPU radi asinhrono: program mu preda posao i odmah nastavi. Ako štoperica ne sačeka da GPU završi, meri samo koliko traje predaja posla. Rezultat izgleda neverovatno brzo, a pogrešan je. Ako je GPU bio zauzet duže nego što je štoperica izmerila, merenje ne valja.

**Rezultat:** GPU vreme nikad nije veće od izmerenog, pa štoperica radi kako treba. Usput je ova provera pokazala gde je usko grlo:

| mreža | koliko je GPU zauzet | šta to znači |
|---|---|---|
| male mreže | 3–56% vremena | GPU uglavnom čeka sledeću naredbu, pa kompajliranje, koje smanjuje broj naredbi, tu pomaže 3–4 puta |
| FFNNBig, posle kompajliranja | 98% vremena | GPU je potpuno iskorišćen, pa dalje ubrzanje može doći samo od manje računanja (afini slojevi), ne od manje naredbi |

### 2.4 Ponavljanje u drugoj sesiji

**Šta:** Ceo deo lestvice ponovio sam u novoj Colab sesiji na istom tipu mašine (A100-40GB), a GPU korake još jednom u trećem procesu.

**Zašto:** Jedno merenje može biti slučajno dobro ili loše: druga mašina iste vrste, drugo opterećenje, drugačije stanje keša. Bez ponavljanja ne znamo koliko je broj stabilan.

**Rezultat:**

- **Koraci do kompajliranog fita** se slažu u okviru 11%; druga mašina je bila oko 5% brža svuda podjednako.
- **Kompajlirani sigmoid** je varirao do 19%. Uzrok sam našao: brzina kompajliranog koda zavisi od toga za koje oblike podataka je kompajliran. U prvom runu je mreža 5layer koristila verziju kompajliranu ranije za manju mrežu i bila sporija (5,8 ms). Kad je kompajlirana baš za sebe, dva puta zaredom je dala 4,6–4,7 ms.
- **Za prezentaciju:** konačno ubrzanje zato pišem kao raspon (oko 2.000–2.400 puta), a ne kao jedan broj.

### 2.5 Da li su tri slike dovoljne

**Šta:** Izmerio sam koliko se vreme razlikuje od slike do slike i od ponavljanja do ponavljanja.

**Zašto:** Originalni eksperiment koristi 1.000 slika, a lestvica 3. Treba pokazati da to za brzinu nije bitno.

**Rezultat:** Vreme varira oko 1% između slika i oko 1,6% između ponavljanja. Za brzinu je broj slika nebitan. Za tvrdnje o širini granica nije, pa ih iz lestvice i ne izvodim.

### 2.6 Kompajliranje bez keša

**Šta:** Drugu sesiju sam pokrenuo sa praznim kešom kompajlera i izmerio koliko traje prvo kompajliranje.

**Zašto:** U prvom runu keš je bio „zagrejan“ ranijom probom, pa je kompajliranje izgledalo jeftino (3,5 s). Ko pokrene kod prvi put, platiće pun trošak.

**Rezultat:** Bez keša kompajliranje traje 13–19 s. U odnosu na verziju bez kompajliranja, isplati se posle oko 500–640 prolaza na 5layer i oko 170 na FFNNBig sa 64 splita. Za veliki eksperiment se isplati, za jedan brz upit ne.

### 2.7 Brojanje GPU kernela

**Šta:** Prebrojao sam koliko zasebnih GPU operacija (kernela) pokreće jedan prolaz.

**Zašto:** Priča o kompajliranju tvrdi da je usko grlo broj malih operacija. To treba izmeriti, a ne pretpostaviti.

**Rezultat:**

- **Smanjenje:** na FFNNBig sa 64 splita broj pada sa 4.412 na 224 po prolazu. To se slaže sa ranijim, nezavisnim merenjem (2.229 → 103, sa duplo manje delova obrade).
- **Greška nađena usput:** prvo merenje je pokazalo 0 kernela svuda. Umesto da prepišem stari broj, napravio sam malu dijagnostiku i našao uzrok: ako se profiler u istom procesu prvo pokrene samo za CPU, posle više ne beleži GPU. Ispravljeno je, i tek onda je izmereno.

### 2.8 Poređenje sa jakim CPU-om

**Šta:** Celu lestvicu sam pokrenuo i samo na M1 Pro procesoru, bez GPU-a.

**Zašto:** Colabov procesor je spor. Ubrzanje u odnosu na spor procesor izgleda veće nego što jeste. Pošteno pitanje glasi: koliko je GPU brži od dobrog procesora koji pokreće isti optimizovani kod?

**Rezultat:**

- **Original na M1** je oko 3 puta brži nego na Colabu.
- **Moje optimizacije pomažu i bez GPU-a:** na M1 daju 103 puta na 5layer.
- **Najbolja GPU verzija prema najboljoj CPU verziji:** 5–7 puta brža na malim mrežama i oko 31 put na FFNNBig.
- **Kako to reći:** „2.000 puta“ je tačno, ali u odnosu na original na sporom Colab procesoru. To se mora reći uz broj.

## 3. Ispravnost: da li su granice i dalje tačne

### Šta znači „ispravna granica“ i zašto se ne može direktno proveriti

Pasado tvrdi: „za svaku jačinu izmaglice u zadatom opsegu, izvod izlaza mreže nije veći od ovog broja“. Da bismo to potpuno proverili, trebalo bi isprobati beskonačno mnogo jačina izmaglice. To nije moguće.

Zato koristim više provera koje napadaju problem iz različitih uglova. Nijedna sama nije dokaz, ali zajedno čine grešku vrlo malo verovatnom.

### 3.1 Uzorkovanje (osnovna provera)

**Šta:** Na 4.001 ravnomerno raspoređenoj tački u opsegu izmaglice stvarni izvod se računa automatskom diferencijacijom. Granica mora biti iznad najvećeg izmerenog.

**Zašto:** Ovo je najjednostavniji način da se nađe kontraprimer. Ako granica ikad padne ispod stvarne vrednosti, ovde će se verovatno videti.

**Rezultat:** U sva tri runa nijedna granica nije ispod uzorkovanog maksimuma.

### 3.2 Gusta pretraga sa doradom

**Šta:** Umesto 4.001 tačke uzeo sam 100.001. Oko 20 najviših vrhova dodatno sam pretražio još gušće.

**Zašto:** 4.001 tačka može da promaši uzak vrh između dve tačke. To je bitno, jer je granica na nekim mestima samo 0,004% iznad uzorkovanog maksimuma. Promašen vrh od 0,01% bio bi nevidljiv, a granica bi bila pogrešna.

**Rezultat:** Gušća pretraga je našla maksimum najviše 0,0024% veći od osnovne. Najmanji razmak granice i stvarnog maksimuma je 1,000039 (granica je 0,0039% iznad). Prvobitno uzorkovanje je bilo dovoljno, ali sada to znamo, a ne pretpostavljamo.

### 3.3 Poređenje sa dve nezavisne metode

**Šta:** Za iste slike i iste opsege izračunao sam granice i sa dve starije metode: intervalnom aritmetikom i običnim zonotopom.

**Zašto:** Intervalna aritmetika je gruba, ali jednostavna i pouzdana. Pasado mora da bude ispod nje, jer bi u suprotnom bio lošiji od najgrublje metode, što ukazuje na grešku. Sve tri metode moraju biti iznad stvarnog maksimuma. Tako se granica ograđuje sa obe strane.

**Rezultat:**

- **Intervalna metoda:** Pasado je ispod nje u svih 48 slučajeva, a sve metode su iznad stvarnog maksimuma.
- **Obični zonotop:** Pasado nije uvek bolji od njega. U 3 od 48 slučajeva je gori za 0,02–0,63%, i to važi i za originalni kod, pa nije posledica mojih optimizacija. Rad to ni ne tvrdi, ali je dobro znati ako se pita.

### 3.4 Greška aproksimacije D za ravan koja je stvarno korišćena

Ovo je najvažnija provera, pa je objašnjavam detaljnije.

**Pozadina:** Na svakom neuronu Pasado zamenjuje krivu ravninom i dodaje rezervu D, najveće odstupanje krive od ravni. Moje optimizacije menjaju način na koji se ravan računa: gelsd, zatvorena forma, kompajlirana verzija. Moj argument da to ne kvari ispravnost glasi ovako: D se uvek računa za ravan koja je stvarno izašla, pa i lošija ravan samo poveća rezervu, a ne pravi grešku.

**Šta:** Ovaj argument nisam hteo samo da tvrdim. Za svih 112.704 neurona, iz tri mreže i tri načina računanja ravni, izračunao sam stvarno najveće odstupanje krive od tačno one ravni koju je kod koristio. Uporedio sam ga sa D koji je kod izračunao. Kutije koje su izgledale sumnjivo ponovo sam izračunao sa 60 decimalnih cifara.

**Zašto 60 cifara:** Odstupanje je razlika dva skoro jednaka broja (oko 0,99 minus oko 0,99). U običnoj preciznosti takva razlika gubi oko 10 cifara tačnosti. Proveravati takav račun istom preciznošću znači porediti grešku sa greškom.

**Rezultat:**

- **D je ispravan do sitnog zaokruživanja.** Najveći manjak je oko 2 jedinice zaokruživanja veličine brojeva od kojih se D računa, isto koliko je izmereno ranije, nezavisno.
- **Postojeći prototip sa zaokruživanjem naviše** pokriva svaku kutiju.
- **Usput je otkrivena zamka u samom merenju.** Kad se manjak meri u odnosu na D, izgleda kao 10⁻⁹, što zvuči ozbiljno. Kad se meri u odnosu na veličinu brojeva koji se oduzimaju, vidi se da je to obično zaokruživanje. Moja prva provera je zbog toga pala, a tek provera sa 60 cifara je pokazala da je problem bio u mom merilu, ne u kodu.

### 3.5 Iste granice na CPU-u i GPU-u

**Šta:** Iste granice izračunao sam na M1 procesoru i uporedio ih sa onima sa A100 grafičke kartice.

**Zašto:** GPU računa drugačijim redosledom i drugim operacijama. Greška specifična za GPU pokazala bi se kao razlika.

**Rezultat:** Granice se slažu do 4 × 10⁻¹² relativno, odnosno do zaokruživanja.

### 3.6 Postojeći testovi i sweep od 264 granice

**Šta:** Pokrenuo sam sve postojeće testove projekta: originalne testove iz uzvodnog repozitorijuma, jedinične testove i test ispravnosti. Uz to sam ponovo izračunao 264 granice iz ranijeg merenja i uporedio ih sa arhivom i sa uzorkovanim maksimumom.

**Zašto:** Da se potvrdi da trenutni kod radi isto kao kod na kome su rađene ranije provere. Kod transformera se od tada nije menjao, a ovo to potvrđuje i u praksi.

**Rezultat:** Sve prolazi, a 264 granice su iste kao u arhivi.

### 3.7 Namerno ubačene greške (mutacioni test)

**Šta:** Napisao sam 33 realistične greške, svaku kao jednu malu izmenu koda. Primeri:

- pogrešan znak;
- zaboravljena apsolutna vrednost;
- pogrešan ugao kutije;
- ravan sa pogrešnim odsečkom;
- rezerva D koja se ne upiše.

Neke od njih su tačno greške koje su stvarno postojale u originalnom kodu, na primer negacija koja ne menja generatore. Za svaku grešku posebno pokrene se ceo skup testova, i bar jedan test mora da padne.

**Zašto:** Test koji prolazi ne znači ništa ako ne bi pao ni na pogrešnom kodu. Ovo je jedini način da se proveri sama provera.

**Rezultat:** Svih 33 greške su uhvaćene, i na procesoru i na GPU-u.

**Mutant M30** posebno pokazuje argument iz 3.4. On namerno daje pogrešnu ravan. Hvataju ga testovi koji proveravaju samu ravan, a test ispravnosti ne pada. To je očekivano: pogrešna ravan daje širu, ali i dalje ispravnu granicu.

## 4. Greške koje su provere otkrile usput

Provere ne služe samo da potvrde. Ove su otkrile pravi problem, i svaki je ispravljen pre nego što je ušao u prezentaciju:

| šta je nađeno | koja provera | posledica da nije nađeno |
|---|---|---|
| izveštaj je za jednu ćeliju usrednjavao pogrešan skup parova | preračunavanje iz sirovih merenja (2.1) | pogrešna granica u tabeli |
| profiler nije beležio GPU posle CPU profila | brojanje kernela (2.7) | broj kernela bi bio prepisan iz starog merenja |
| kompajlirani sigmoid zavisi od istorije kompajliranja | ponavljanje u drugoj sesiji (2.4) | jedan broj umesto raspona, oko 19% nepreciznosti |
| zastareli keš kompajlera na Macu je tiho isključivao kompajliranje | testovi kompajlirane putanje (3.6) | lokalni testovi bi lažno padali |
| dva mutanta sa istim imenom su prepisivala rezultate | moj pregled tabele mutacionog testa (3.7) | rezultati 5 mutanata bi se izgubili |
| pogrešno merilo za grešku D | provera sa 60 cifara (3.4) | lažna uzbuna, ili pogrešno odbačena tačna tvrdnja |

Poslednje dve greške su bile moje, u samim proverama. Zato je svaki neočekivan rezultat bio istražen do uzroka, a ne prihvaćen ili odbačen na prvi pogled.

## 5. Šta ove provere ne pokrivaju

Ovo treba reći otvoreno:

1. **Testiranje nije dokaz.** Provereno je 4 mreže, 3 slike, 2 jačine izmaglice i do 64 splita. Nema razloga da se druge slike ponašaju drugačije, ali to nije dokazano.
2. **Zaokruživanje u računaru.** Rezerva D može biti manja od tačne vrednosti za oko 2 jedinice zaokruživanja. U strogo matematičkom smislu granica zato nije garantovana u aritmetici računara. Isto važi i za originalni Pasado. Ovo je predlog za sledeći korak: formalna granica sa zaokruživanjem naviše, za koju prototip već postoji i prolazi sve kutije.
3. **Matematika metode** se oslanja na objavljeni rad i na ispravku `b526762`. Nisam je ponovo dokazivao.
4. **Mutanti hvataju samo zamišljene greške.** Greška kakvu niko nije zamislio može da prođe, iako 33 raznovrsne greške čine to manje verovatnim.
5. **Test ispravnosti proverava samo sekvencijalnu putanju koda.** Batched putanju pokrivaju testovi koji je porede sa sekvencijalnom, kao i provere 3.2 i 3.4. Pokrivena je, ali posredno.

## 6. Kako se sve ovo ponavlja

Sve skripte su u repozitorijumu, a rezultati u `results/`:

```
sh experiments/run_all_checks.sh                         # svi testovi, 264 granice, mutacioni test
python experiments/verify_ladder_claims.py all           # provere 2.2, 2.3, 3.2, 3.3, 3.4
python experiments/verify_d_shortfall.py                 # studija zaokruživanja D
python experiments/mutation_check.py --device cuda       # mutacioni test na GPU-u (Colab)
python experiments/measure_speedup_ladder.py --device cuda --out <folder>   # lestvica
```

Folderi sa rezultatima:

- [`../ladder_a100_40gb/`](../ladder_a100_40gb/README.md): glavna lestvica;
- [`../ladder_a100_40gb_rep2/`](../ladder_a100_40gb_rep2/summary.md): druga sesija;
- [`../ladder_kernels/`](../ladder_kernels/summary.md): brojanje kernela;
- [`../ladder_m1pro_cpu/`](../ladder_m1pro_cpu/summary.md): samo CPU;
- [ovaj folder](README.md): sve provere ispravnosti.
