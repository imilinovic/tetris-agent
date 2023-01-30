# Tetris Agent

## Opis agenta
Tetris agent koji za igranje koristi 5x5x1 feedforward neuronsku mrežu sa hidden layerom koji koristi sigmoid kao aktivacijsku funkciju. Agent igru igra tako da svaki mogući potez ocijeni neuronskom mrežom te odabere onaj minimalni. Neuronska mreža (odnosno njeni parametri) se trenira genetskim algoritmom. Kao crossover se koristi SBX (Simulated Binary Crossover). 

Svaki parametar novog djeteta ima vjerojatnost p=0.05 da bude mutiran. Iz svake generacije se sačuva 20% najboljih kromosoma te se od njih izgradi ostatak generacije. Kromosomi se ocjenjuju (fitness function) tako da se s njegovim parametrima odigraju 3 partije tetrisa i uzme prosječan score. 

Svakih 10 generacija se smanjuje parametar mutation_rate koji određuje koliko će mutiranje mijenjati parametre. Ako u 30 generacija ne dođe do promjene najboljeg fitnessa ponovno se inicijaliziraju svi osim 3 najbolja kromosoma. 

## Korištenje
Potrebno je instalirati python librarye iz requirements.txt.

Program se može pokrenuti sa 3 različita parametra (argparse)

--load {ime_filea} - učitaj weightove iz .npz filea, inače ih random inicijaliziraj, nakon svake generacije se weightovi (sortirani od najboljeg prema najgorem (po fitnesu)) te generacije spremaju u weights_gen_{id_generacije}.npz \
--gui - koristi tetris gui, ovo bi se trebalo koristiti samo uz --load kako bi se vizualizirala strategija agenta jer je znatno sporije \
--mutation-step {float} - postavi inicijalni mutation_step na neki broj, po defaultu je 0.2 (note: smanjuje se svakih 10 generacija) 

U fileovima good.npz, good2.npz, good3.npz nalaze se neki dobri weightovi dobiveni u 50-tak generacija.

Sav ispis se sprema u logs.log (no ne brišu se stari logovi nego se samo ispis appenda pa ga je potrebno ponekad ručno obrisati)