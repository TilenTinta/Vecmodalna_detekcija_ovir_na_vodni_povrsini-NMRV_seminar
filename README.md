# Večmodalna detekcija ovir na vodni površini

Vecmodalna_detekcija_ovir_na_vodni_povrsini-NMRV_seminar

Avtonomna vožnja je področje, ki v zadnjem času doživlja velik razvoj. Največ govora o njej je predvsem v cestnem prometu, vendar pa ni prisotna le v avtomobilski industriji. Raziskave in prve implementacije se pojavljajo tudi v pomorstvu, kjer se avtonomna plovba sooča z drugačnimi izzivi zaradi spremenljivih okolij, značilnih za pomorstvo.

Detekcija ovir na vodi je poseben problem, ki se razlikuje od detekcije na kopnem. Vodna površina se stalno spreminja zaradi vplivov okolja, kot so na primer valovanje, bleščanje, odsevi in megla, ki se na vodi pogosto pojavlja. Poleg tega je razvoj podatkovnih zbirk za ta namen še v zgodnji fazi. Zaradi tega je učenje robustnih modelov strojnega učenja veliko težje v primerjavi s cestnim prometom, kjer so podatkovne zbirke dobro anotirane in številčne.

V tem članku predstavljamo več pristopov za zaznavanje in segmentacijo ovir na vodni gladini z uporabo različnih vhodnih oblik podatkov. Osredotočamo se na uporabo barvnih in termalnih slik ločeno, zajetih z dvema neodvisnima kamerama.

V prvem delu predstavljamo semantično segmentacijo z uporabo arhitekture SegFormer, kjer rezultate primerjamo s klasičnim modelom U-Net. V drugem delu pa se osredotočimo na detekcijo ovir s pomočjo modela YOLO11, prilagojenega za delo z enim razredom, ki predstavlja dinamične ovire.

Skozi analizo različnih arhitektur in vhodnih podatkov raziskujemo, kako vrsta vhodnih podatkov vpliva na natančnost zaznave. Pokazali bomo, da barvne slike pri dnevnih pogojih prekašajo termalne, zlasti na področju detekcije premikajočih se ovir. Naše ugotovitve prikazujejo prednosti in omejitve posameznih pristopov pri segmentaciji in detekciji ovir na vodi.

## Struktura projekta
* `Detekcija_RGB_YOLO/` - Implementacija detekcije na RGB slikah z uporabo YOLO11 modela.
* `Detekcija_RGB_YOLO_ImgTile/` - Implementacija detekcije na razrezanih RGB slikah z uporabo YOLO11 modela.
* `Detekcija_Thermal_YOLO/` - Implementacija detekcije na termalnih slikah z uporabo YOLO11 modela.
* `Segmentacija_RGB_U-Net/` - Koda za segmentacijo RGB slik z uporabo modela U-Net.
* `Segmentacija_RGB_SegFormer/` - Koda za segmentacijo RGB slik z uporabo modela SegFormer.
* `Segmentacija_Thermal_U-Net/` - Segmentacija termalnih slik z uporabo U-Net.
* `Segmentacija_Thermal_SegFormer/` - Segmentacija termalnih slik z uporabo SegFormer.
* `NMRV_viewer.py` - Skripta za vizualizacijo učnih primerov segmentacije in detekcije.

## Zahteve
* Python 3.x
* Knjižnice: OpenCV, PyTorch, YOLO11.
