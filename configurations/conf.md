# Descrizione esperimenti

## 1. Esperimento: Parete Multistrato (Radioprotezione Standard)

Questo esperimento simula la classica configurazione di scudi protettivi utilizzati nei laboratori nucleari. L'obiettivo è bloccare un fascio direzionale di neutroni sfruttando tre materiali con proprietà fisiche complementari.

- **Dinamica Fisica:** Il neutrone incontra prima uno spesso blocco di **Ferro**, il cui scopo è abbattere l'energia cinetica dei neutroni più veloci tramite duri urti inelastici. Successivamente entra nell'**Acqua** (il moderatore per eccellenza), dove le collisioni con i nuclei di idrogeno rallentano la particella fino a energie termiche. Infine, i neutroni ormai lenti sbattono contro il **Boro**, che possiede una sezione d'urto di assorbimento altissima e agisce da "spugna" finale.
- **Cosa testa nel vostro codice:** Verifica l'accuratezza del tracciamento lungo una linea prevalentemente retta e, soprattutto, che le probabilità di interazione (sezioni d'urto) vengano applicate correttamente nel giusto ordine geometrico.

## 2. Esperimento: Cubo Moderatore a "Matrioska" (Isotropia e Termalizzazione)

Questa geometria è il fondamento dell'Analisi per Attivazione Neutronica (NAA). Simula una sorgente puntiforme immersa al centro di un massiccio moderatore solido per creare una "nube" di neutroni termici.

- **Dinamica Fisica:** La particella nasce nel vuoto e viaggia indisturbata finché non colpisce le pareti interne della camera in **Grafite**. Da quel momento, inizia un percorso a zig-zag casuale tridimensionale (random walk), perdendo una frazione di energia a ogni rimbalzo. Non essendoci materiali assorbenti, il neutrone continua a deviare finché non esce dai bordi esterni del sistema o non viene ucciso dal limite massimo di step della simulazione.
- **Cosa testa nel vostro codice:** È il collaudo perfetto per la vostra matematica di generazione degli angoli di scattering 3D. Se l'isotropia del codice è corretta, la distribuzione spaziale dei neutroni all'interno della grafite dovrà risultare perfettamente sferica.

## 3. Esperimento: Labirinto di Streaming (Dispersione nei Bunker)

Una geometria progettata per calcolare le fughe parassite di radiazioni ("streaming") lungo i corridoi degli acceleratori o delle sale di radioterapia. I neutroni non amano le linee rette; questo esperimento testa se riescono a "svoltare l'angolo".

- **Dinamica Fisica:** I neutroni vengono sparati in un tunnel a forma di "L". Le pareti in **Calcestruzzo** tendono ad assorbire la maggior parte delle particelle, ma una percentuale rimbalza sulle pareti (effetto albedo). Solo i neutroni che subiscono la sequenza geometricamente perfetta di rimbalzi riescono a superare la curva e raggiungere l'uscita.
- **Cosa testa nel vostro codice:** Mette alla prova la robustezza del codice nelle geometrie con vuoti complessi e verifica se il motore gestisce correttamente le condizioni di perdita di particelle (leakage) dai bordi aperti della geometria.

## 4. Esperimento: Collimatore a Fessura (Fascio ad Alta Precisione)

Questo setup riproduce l'estrazione di un fascio neutronico da un reattore di ricerca, utilizzato per scansionare la struttura reticolare dei materiali (diffrazione neutronica).

- **Dinamica Fisica:** Un microscopico canale di vuoto lungo 40 cm è stretto in una morsa di spessi blocchi di **Cadmio**. Il cadmio è un "veleno" formidabile che fagocita immediatamente quasi ogni neutrone che lo tocca. Di conseguenza, solo i neutroni con una traiettoria millimetricamente parallela all'asse del tunnel sopravvivono.
- **Cosa testa nel vostro codice:** È uno *stress test* per la precisione in virgola mobile (ray-casting rasente le superfici) e per il meccanismo di annichilazione del codice (sezione d'urto di assorbimento puro estrema).

## 5. Esperimento: Reticolo del Nocciolo (Simulazione di Reattore)

Una riproduzione in miniatura (su una griglia 3x3) della topologia interna del nocciolo di un reattore ad acqua pressurizzata, dove il combustibile e il moderatore sono disposti a scacchiera.

- **Dinamica Fisica:** Sottili "barre" di **Uranio** sono separate tra loro da canali di **Acqua**. I neutroni nascono e viaggiano attraversando continuamente i due materiali: subiscono scattering elastico nell'acqua e assorbimento (che in una simulazione più avanzata innescherebbe una nuova fissione) nell'uranio.
- **Cosa testa nel vostro codice:** Misura l'efficienza computazionale nuda e cruda della GPU. A causa delle dimensioni ridotte delle barre, le particelle cambiano regione e materiale decine di volte in pochissimi microsecondi, obbligando il kernel CUDA a risolvere un numero enorme di intersezioni (boundary crossings) in uno spazio densamente popolato.
