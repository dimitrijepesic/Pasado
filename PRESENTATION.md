# Internship checkpoint — Pasado

Tekst za Canvu je na engleskom. Govor je na srpskom. Za 10–12 minuta koristi glavnih 13 slajdova, a detalje o solverima, kompajlerima i proverama drži u rezervi.

## 1. Pasado — Accelerating Neural Network Sensitivity Analysis

**Internship Checkpoint**  
Dimitrije Pešić · University of Belgrade · November 2026

**Govor.** „Zanimalo me je koliko može da se ubrza računanje užih granica izvoda. Krenuo sam od postojećeg Pasado benchmarka i menjao pristup kako su merenja otkrivala sledeće usko grlo.“

## 2. The Work Repeated at Every Sigmoid Neuron

- Fit a plane to the joint derivative update `σ′(x) · y` on a **5 × 5 grid**.
- Bound the absolute residual over the input box by checking corners and solving for stationary points on the edges.
- **FFNNBig: 4,096 fits and error checks per pass at one split.** The paper reports **1.49 s vs. 0.361 s** for standard zonotope AD.

*Original paper, mean runtime per input configuration.* [§4.1 and §5.4](https://misailo.cs.illinois.edu/papers/pasado-oopsla23.pdf)

**Govor.** „Za kontekst, ovo je korak koji se ponavlja kroz celu priču. Ulaz u sigmoid je x, a y je njegov izvod po izmaglici. Fit biram na 25 tačaka. Grešku računam odvojeno na celoj kutiji. Za stacionarne tačke na nelinearnim ivicama rešavam kubnu jednačinu. Zato imamo hiljade malih fitova i provera po prolazu.“

## 3. Profiling Changed the CPU Plan

- **Small networks:** error checks took **53–54%** of a pass. Vectorizing them gives **2.0–2.1×**.
- **All networks:** one batched `gelsd` per layer instead of one per neuron. **4.1–4.4×** more on the small networks, **2.9–3.2×** on FFNNBig.
- **CPU total on fixed code:** **9.6–10.7×** on the small networks and **5.5–5.8×** on FFNNBig.

*Post-fix speedup ladder, host CPU of the A100 machine (Xeon 2.2 GHz, 6 threads), float64, `17ad4b0`. Earlier pre-fix i5 runs gave 3.24–3.64× and 1.65× before the batched grid.* [Ladder](results/ladder_a100_40gb/README.md) · [Pre-fix CPU results](results/optimization_section_5_4/FORMAL_SUMMARY.md)

**Govor.** „Prvi rezultat od 1,87 puta sam odbacio kad sam utvrdio da su se uslovi menjali između merenja. Uveo sam naizmenične probe i pinovanje na performansna jezgra. Tek onda sam koristio parove iz iste sesije. Profil male mreže mi je pokazao gde da uklonim petlje, a profil velike zašto vredi probati GPU.“

## 4. Why the First GPU Version Was Hybrid

- **CUDA `gels`:** close agreement on full-rank inputs, large coefficient discrepancies on rank-deficient inputs.
- **Decision:** keep batched `gelsd` on the CPU. Move dense matmul and tensor checks to the GPU.
- **FFNNBig, one split:** **347 → 68 ms (5.1×)**. The CPU solve still took about **49%** of the pass.
- **Small networks:** the hybrid was **slower** than the CPU (**0.64–0.73×**). The work per neuron is too small for the GPU.

*Driver comparison: A100-80GB, 2,000 synthetic fits. Timing: post-fix ladder, A100-40GB and its host CPU, `17ad4b0`.* [Driver test](results/optimization_section_5_4/GPU_READINESS.md#phase-12-rank-deficient-driver-test---gels-confirmed-unsafe) · [Ladder](results/ladder_a100_40gb/README.md)

**Govor.** „Jacobovo zapažanje ima smisla za pun rang. Na 1.200 takvih fitova najveća razlika u koeficijentima bila je 6,95 × 10⁻¹². Na 800 rang-deficitnih fitova svaki je prešao prag od 10⁻⁶ u poređenju sa CPU referencom. Zato sam zadržao gelsd za fit i izmerio šta GPU donosi ostatku analize.“

## 5. October: Checking the Bounds Changed the Project

- **Negation:** flip the generators along with the center to preserve shared dependencies.
- **Approximation error:** check `|g − P|` at boundary extrema to cover negative residuals.
- **Validation:** exact haze endpoints, targeted regression tests, **264/264** bounds above sampled maxima, and all **33** planted bugs caught on CPU and CUDA.

*Fix `b526762`, October 22. Final test run on `91ebd88`.* [Regression tests](experiments/test_soundness.py) · [Final checks](CHECKPOINT.md#6-checks-on-this-machine)

**Govor.** „Upoređivanje izlaza i provera prema autogradu otvorili su posebno pitanje korektnosti. Pogrešna negacija menja putanju haze ulaza, čak i ako su rasponi pojedinačnih piksela širi. Provera sa znakom može da propusti najveće odstupanje ispod ravni. Prva greška pogađa i običan zonotope baseline, pa sam nakon ispravke ponovio poređenje. Uzorkovanje služi da pronađem kontraprimer, a ciljanim testovima proveravam konkretan mehanizam greške.“

## 6. Batched Splits Exposed the Next Bottleneck

- Split the haze interval to tighten the bound. Batch the subintervals to process them together.
- **64 splits:** CPU `gelsd` took **84%** of the 2,318 ms batched pass on FFNNBig, and **88–90%** on the small networks.
- Revisit a closed-form fit now that the CPU solver dominates the GPU workload.

*Post-fix ladder, A100-40GB, float64, `17ad4b0`.* [Ladder](results/ladder_a100_40gb/README.md) · [Earlier split sweep](results/gpu_hybrid_batched_splits_2026-10-24/README.md)

**Govor.** „Na CPU-u sam odložio zatvorenu formu zbog numeričkih pitanja i postojećeg batched solvera. Na GPU-u su se uslovi promenili. Brzo izvršavanje ostalih koraka ostavilo je CPU regresiju kao dominantan trošak. Tu se isplatilo vratiti strukturisanom fitu.“

## 7. Choosing a GPU Replacement for CPU Least Squares

| Method | Median fit time |
|---|---:|
| CPU `gelsd` with transfers | 565.13 ms |
| GPU pseudoinverse | 35.38 ms |
| Closed-form fit with CPU fallback | 3.89 ms |
| **Closed-form fit with numerical rank handling** | **3.83 ms** |
| Closed-form fit with exact degeneracy checks | 1.62 ms |
| Sigmoid-specific closed-form fit | 1.25 ms |

**Selected under the main protocol:** numerical rank handling on the GPU, without host synchronization.

*A100-40GB · float64 · solver only · 65,536 fits · `832323a`.* [Timing](results/lstsq_comparison_2026-10-27_v3_zpivot/timing_cuda.csv) · [Selection](results/lstsq_comparison_2026-10-27_v3_zpivot/README.md)

**Govor.** „Uporedio sam pet zamena na stvarnim i sintetičkim kutijama. Popravio sam overflow na ekstremnim primerima i osetljivost na redosled sabiranja na uskim kutijama. Najbrža verzija koristi oblik sigmoid izraza. Po glavnom protokolu, izabrana verzija i hibrid bili su u istoj kategoriji po numerici i vremenu. Presudilo je izvršavanje bez host sinhronizacije. Ovo je specijalizovan fit tri koeficijenta na Kartezijevoj mreži, sa obradom numeričkog ranga.“

## 8. What Compilation Added

- **FFNNBig, 64 splits:** **2,318 ms** batched CPU fit → **349 ms** eager GPU fit → **306 ms** compiled fit only → **239 ms** compiled sigmoid.
- **About 1.46× over eager GPU execution on FFNNBig**, and **5.5–5.8×** on the small networks with 64 splits.
- Two attempts stopped at correctness checks. The third passed. CUDA kernels per pass fell from **4,412 to 224**, peak GPU memory from **14.9 to 10.3 GiB**.

*Post-fix ladder, A100-40GB, float64, `17ad4b0`; kernel counts from a profiling rerun of the same steps on the same machine type, `1c74a35`.* [Ladder](results/ladder_a100_40gb/README.md) · [Kernel rerun](results/ladder_kernels/summary.md)

**Govor.** „Najveći pad vremena dolazi kada fit ostane na GPU-u. Širi graf omogućava kompajleru da spoji i operacije oko fita. Prva verzija je pala na ponovljivosti i toleranciji za generatore. Druga je popravila promenu grafa između slojeva, ali dijagnostički graf nije dao iste bitove kao graf koji merimo. U trećoj sam proveravao veličine iz istog grafa. Na velikoj mreži dobitak je oko 1,46 puta preko eager GPU verzije, jer tamo već dominiraju afini slojevi. Na malim mrežama, gde je sigmoid i dalje glavni trošak, dobitak je 5,5 do 5,8 puta.“

## 9. One Machine, Every Step

| Step | 5layer, 64 splits | FFNNBig, 64 splits |
|---|---:|---:|
| Original Pasado (CPU) | 11,510 ms | 128,110 ms |
| Vectorized checks | 2.05× | 1.9× |
| Batched `gelsd` | 8.7× | 5.6× |
| Real cubic | 10.1× | 5.5× |
| GPU hybrid | 7.4× | 31× |
| Batched splits | 53× | 55× |
| Closed-form fit | 339× | 367× |
| Compiled fit | 605× | 418× |
| **Compiled sigmoid** | **1,976× (5.8 ms)** | **537× (239 ms)** |

- **Speedup over the original**, every step on the same machine: steps up to the real cubic on its CPU, the rest on its GPU.
- **Split of the gain:** about **10×** from the CPU algorithm, **195×** more from the GPU (FFNNBig: **5.5×** and **97×**).
- **Soundness:** every bound stays above the sampled maximum. Bounds differ by at most **1.1 × 10⁻⁴** relative, from the real cubic.
- **Repeated in a second session:** steps up to the compiled fit within **11%**. The compiled sigmoid takes **4.6–4.7 ms** on 5layer when compiled for its own shapes (**about 2,360×**); the 5.8 ms above reused a graph compiled for 3layer.

*Post-fix, float64, Xeon 2.2 GHz (6 threads) + A100-SXM4-40GB, torch 2.14.1, 3 images × 2 ε, median of 5 passes, `17ad4b0`. FFNNBig CPU steps at 64 splits: fewer passes.* [Ladder](results/ladder_a100_40gb/README.md) · [All 12 configurations](results/ladder_a100_40gb/summary.md)

**Govor.** „Raniji brojevi su dolazili sa različitih mašina, pa sam sve korake ponovo izmerio na jednoj, od originalnog koda do kompajliranog sigmoida. Koraci do realne kubne formule idu na CPU-u te mašine, ostali na njenom GPU-u. Na 5layer sa 64 splita prolaz pada sa 11,5 sekundi na 5,8 milisekundi. U drugoj sesiji, kad je graf kompajliran baš za tu mrežu, poslednji korak je 4,6 milisekundi, pa je ukupno ubrzanje između 2.000 i 2.400 puta. CPU algoritam donosi oko deset puta, a GPU još oko 195. Sam prelazak na GPU na malim mrežama usporava, jer je posao po neuronu premali. Dobitak dolazi tek kad batchujem splitove i izbacim CPU solver. Na FFNNBig posle kompajliranja 76 posto vremena odlazi na afine slojeve, pa je to sledeće usko grlo.“

## 10. How the Results Were Checked

| Check | Result |
|---|---|
| Each step runs the code it claims; the timer waits for the GPU | 9/9 steps, 8/8 profiles |
| Bound vs. a dense search for the true maximum | 72/72 above, min ratio 1.000039 |
| Pasado vs. interval AD | never looser, 48/48 |
| Plane error D, rechecked in 60-digit arithmetic | 112,704 boxes, at most 2 ε·T short |
| Same bounds on M1 CPU and A100 GPU | within 4.2 × 10⁻¹² relative |
| 33 planted bugs, on CPU and on CUDA | 33/33 caught |

**Testing, not proof.** The D shortfall is floating-point rounding, also present in the original Pasado; outward-rounded D covers every box.

*M1 Pro CPU and Colab A100, November 15–16, `f89a5d5`.* [Checks](results/verification_2026-11-15/README.md) · [Kako i zašto (srpski)](results/verification_2026-11-15/IZVESTAJ.md)

**Govor.** „Pošto su brojevi brzine i granica glavni rezultat, posebno sam proverio da li su tačni. Prvo, da svaki korak lestvice zaista izvršava kod za koji kažem da ga izvršava i da štoperica čeka GPU. Drugo, granica mora da bude iznad stvarnog maksimuma izvoda: pored uzorkovanja od 4.001 tačke, uradio sam gustu pretragu sa 100.001 tačkom i doterivanjem oko vrhova, i svih 72 granica su iznad, najtešnje za 0,004 posto. Treće, Pasado nikad nije širi od intervalne aritmetike; od običnog zonotopa je u 3 od 48 slučajeva malo širi, do 0,63 posto, isto i u originalu. Četvrto, za grešku aproksimacije D sam za svaku od 112.704 kutije izračunao tačan maksimum u aritmetici sa 60 cifara. D je ispod njega najviše dve jedinice zaokruživanja, ista floating-point pojava kao u originalu, i to je razlog za numerički pravac na kraju. Peto, iste granice na M1 CPU-u i na A100. Na kraju, ubacio sam 33 namerne greške u kod i testovi su uhvatili svaku, i na CPU-u i na CUDA. Ovo je testiranje, ne dokaz.“

## 11. What Is Measured and What Comes Next

- **Measured:** every step on one A100 machine, post-fix. Whole-sigmoid Inductor on A100. Local CPU runs with Inductor, JAX/XLA and TVM.
- **Pending:** same-GPU auto_LiRPA comparison and A100 compiler matrix.
- **Follow-up:** the affine layers take **66–76%** of FFNNBig after compilation. Run the eight-layer depth study.

[Ladder](results/ladder_a100_40gb/README.md) · [Checkpoint status](CHECKPOINT.md#4-still-pre-fix-or-not-measured) · [Notebooks](colab/README.md)

**Govor.** „Na A100 sam završio celu lestvicu ubrzanja na jednoj mašini i konkretno Inductor merenje. Sva tri alata sam probao lokalno na CPU-u. TVM fit je stao pri uvozu, a drugi kernel je prošao uz promenu izlaznog tipa matmulta. Neptune sam procenio prema obrascu redukcija i nisam ga pokrenuo. Sledeće bih uradio poređenje sa auto_LiRPA na istom GPU-u. Na velikoj mreži sledeći kandidat za ubrzanje su afini slojevi, a ne sigmoid.“

## 12. Re-evaluating Section 5.4

| Network | Archived pre-fix artifact | Local post-fix rerun |
|---|---:|---:|
| 3layer | 10.4× | 3.0× |
| 4layer | 34.2× | 4.9× |
| 5layer | 118.8× | 8.1× |
| **FFNNBig** | **181.8×** | **8.8×** |

**The baseline changed most.** On FFNNBig at ε = 0.2, the standard zonotope bound fell from **759,353 to 34,130**. The Pasado bound fell from **4,177 to 3,889**.

*Largest ratio of mean standard-zonotope to Pasado bounds across 16 haze ranges. Local post-fix rerun: 30 images, one split, M1 Pro CPU, float64, seed 7, `91ebd88`. Fig. 8 reports 1,000 images.* [Post-fix results](CHECKPOINT.md#21-section-54-after-the-fix-local-new)

**Govor.** „Ovo je posledica zbog koje je vredelo proveriti obe metode. Na istoj jačini izmaglice brojilac, obična zonotopska granica, pao je oko 22 puta. Pasado granica se smanjila oko sedam posto. Obični zonotope koristi negaciju i na ulazu i u svakom sigmoid sloju, pa je vraćanje zavisnosti imalo veći efekat. Ovo je rezultat lokalnog ponovljenog merenja uz arhivu, a za ažuriranje same Fig. 8 treba dogovoriti ponavljanje njenog protokola.“

## 13. Discussion

- **Section 5.4** — Should we rerun the Fig. 8 protocol and update the artifact or document the correction in the PR?
- **Application direction** — Certify prediction stability under haze using adaptive GPU-batched splits.
- **Numerical direction** — Develop a floating-point error bound for the compiled sigmoid step and measure its cost.

**Govor.** „Prvo bih voleo da dogovorimo kako predstavljamo ispravku iz odeljka 5.4 i da li ponavljamo protokol Fig. 8. Posle stabilizacije koda, moj predlog je da probam sertifikaciju predikcije kroz izmaglicu. Drugi pravac je da izvedem floating-point granicu za lokalnu grešku D. Koji od ta dva rezultata ti deluje važniji kao sledeći korak?“

---

## Backup — pitanja i izvori

- **„Zašto ne CUDA gels?“** Sintetički test, 1.200 slučajeva punog ranga i 800 rang-deficitnih. Najveće apsolutno odstupanje koeficijenata kod punog ranga 6,95 × 10⁻¹². Medijane na rangovima 2 i 1 bile su oko 1,27 × 10¹⁵ i 2,62 × 10³¹. Test poredi koeficijente sa CPU referencom. [Rezultati testa](results/optimization_section_5_4/GPU_READINESS.md) · [Skripta](experiments/colab_rank_deficient_lstsq_test.py)
- **„Da li si napisao svoj gelsd?“** Rešava se fit ravni sa tri koeficijenta na Kartezijevoj mreži. Centrirane kolone su ortogonalne, pa se nagibi računaju odvojeno. Odluka o numeričkom rangu približno prati prag reference. Oko 2 × 10⁻¹³ je najgore relativno odstupanje izlaznih granica u testiranom poređenju na mrežama. To je metrika za granice, a koeficijenti i ponašanje blizu praga ranga provereni su zasebno. Dodatno rangiranje sa svim proverama na CUDA izdvojilo je sigmoid-specifičnu verziju. [Protokol i oba rangiranja](results/lstsq_comparison_cuda_2026-10-27/README.md) · [CUDA end-to-end provera](results/planar_fit_cuda_2026-10-27/README.md)
- **„Šta je zaustavilo v1 i v2?“** V1 je imala razlike u poslednjim bitovima kad bi statički graf zamenio dinamički i prekoračenje tolerancije za dualne generatore. U v2 je dinamička dimenzija popravila ponovljivost. Dijagnostički graf sa dodatnim izlazima promenio je fuziju, pa preduslov za proveru D nije prošao. V3 proverava i meri istu funkciju. [V1](results/compiled_sigmoid_cuda_2026-10-28_gate_stop/README.md) · [V2](results/compiled_sigmoid_v2_cuda_2026-10-29_gate_stop/README.md) · [V3](results/compiled_sigmoid_v3_cuda_2026-10-29/README.md)
- **„Koliko kompajler pomaže na manjoj mreži?“** Na `3layer` sa jednim splitom: 16,74 ms eager GPU fit, 9,57 ms kompajlirani fit, 2,60 ms kompajlirani sigmoid. Odnos prema batched CPU `gelsd` (10,40 ms) je 4,0×. Kompajliranje sa praznim kešom traje 13–19 s, a sa zagrejanim oko 3,5 s. Sa praznim kešom se isplati posle oko 500–640 prolaza na 5layer i oko 170 na FFNNBig sa 64 splita. [Lestvica](results/ladder_a100_40gb/README.md)
- **„Zašto je GPU hibrid sporiji na malim mrežama?“** Sloj male mreže ima malo neurona, pa GPU ne može da se napuni. `gelsd` i dalje ide na CPU-u, pa se dodaju i prenosi. Na FFNNBig isti korak daje 5,1×. Na malim mrežama dobitak dolazi tek od batchovanja splitova (7,2–7,8×) i fita na GPU-u (6,0–6,4×). [Lestvica](results/ladder_a100_40gb/README.md)
- **„Zašto je zatvorena forma sporija sa jednim splitom?“** Sa jednim splitom fitova je malo, pa je `gelsd` jeftin, a zatvorena forma u eager modu pokreće mnogo malih kernela (na 5layer sa 64 splita 2.198 po prolazu, prema 946 za korak sa `gelsd`). Na malim mrežama korak 6 je tada 0,66× koraka 5. Zato `PASADO_AUTO_PLANE_FIT` bira metodu po veličini, sa pragom 512 za float64. [Lestvica](results/ladder_a100_40gb/README.md) · [Crossover](results/plane_fit_crossover_2026-10-28/README.md)
- **„Gde su ostali kompajleri?“** M1 Pro CPU, 65.536 fitova: 21,78 ms eager, 2,07 ms Inductor default, 6,75 ms JAX/XLA. TVM Relax nije uvezao fit. Kod kernela za granice uvezao je float64 matmul sa float32 izlazom. [CPU tabela](results/compilers_2026-11/local_m1pro_cpu/table.md) · [TVM log](results/compilers_2026-11/local_m1pro_cpu/tvm.jsonl) · [Neptune analiza](docs/NEPTUNE_ANALYSIS.md)
- **„Šta pokazuju testovi granica?“** Završni sweep ima 264/264 granica iznad uzorkovanog maksimuma, a zasebna matrica Section 5.4 ima 1.920 takvih granica po metodi. Dokaz za kontinuirani domen i floating-point implementacija ostaju zasebna pitanja. [Završne provere](CHECKPOINT.md#6-checks-on-this-machine) · [Matrica Section 5.4](CHECKPOINT.md#21-section-54-after-the-fix-local-new)
- **„Zašto ova dva pravca?“** Sertifikacija meri koliko slika zadržava klasu pod izmaglicom pri istom vremenskom budžetu. Floating-point pravac počinje od lokalne greške D, gde već postoji empirijski prototip. [DeepCert](https://arxiv.org/abs/2103.01629) · [Jia i Rinard, SAS 2021](https://people.csail.mit.edu/rinard/paper/sas21.exploiting.pdf) · [Outward-rounding prototip](results/outward_rounding_prototype_2026-11/README.md)
- **„Koliko je to brže od dobrog CPU-a?“** Na M1 Pro, sa 64 splita, original traje 3.242 ms na 5layer i oko 43 s na FFNNBig, oko 3 puta brže od Colabovog Xeona. Isti optimizovani kod samo na M1 CPU-u daje 103× na 5layer i 5,9× na FFNNBig. Najbolja A100 verzija je od najbolje M1 CPU verzije brža 5,4–6,8 puta na 5layer i oko 31 put na FFNNBig. Ubrzanje od 2.000× meri se prema originalu na Xeon hostu A100 mašine. [M1 lestvica i poređenje](results/verification_2026-11-15/README.md#cpu-only-ladder-on-the-m1-pro-ladder_m1pro_cpu)
- **„Kako znaš da je sve ispravno?“** Pored testova i 33 namerno ubačene greške koje testovi hvataju: provera da svaki korak izvršava kod koji tvrdim, gusta pretraga maksimuma izvoda, poređenje sa intervalnim i običnim zonotopom, i provera D za ravan koja je stvarno korišćena, sa 60 cifara. D je ispod tačnog maksimuma najviše 2 eps·T zbog zaokruživanja. Formalno to nije dokaz. [Provere](results/verification_2026-11-15/README.md)
