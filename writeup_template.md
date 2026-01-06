## Writeup Template

### You use this file as a template for your writeup.

---

**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration
#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Sociva kamere nisu savrsena i unose distorziju , što uzrokuje da ravne linije na ivicama slike izgledaju zakrivljeno. 
Da bismo precizno merili krivinu puta moramo resiti taj problem. Postupak se nalazi u fajlu calibration.py

Koraci:

    Priprema koordinata: Kreirao sam niz objectPoints koji predstavlja stvarne koordinate uglova sahovske table (npr. 9×5 uglova).

    Koristeci cv.findChessboardCorners, prošao sam kroz sve slike iz camera_cal foldera.

    Za svaku uspešnu detekciju, koristio sam cv.cornerSubPix da dobijem preciznost na nivou sub-piksela.

    Pozivom cv.calibrateCamera, algoritam je uporedio 3D objectPoints i 2D imgPoints da bi izračunao matricu kamere i koeficijente distorzije.

    Rezultate sam sacuvao u .npz fajl da ne bih ponavljao proces pri svakom pokretanju videa.

    ![Calibration Original](https://github.com/misojca/Lane-Lines-Detection/blob/main/result_files/calibration_original.jpg?raw=true)
    

### Pipeline (single images)
#### 1. Provide an example of a distortion-corrected image.

    Primenjena je funkcija cv.undistort

    Bez ovoga radijus krivine bi bio pogresan jer bi distorzija sociva vestacki povecala ili smanjila stvarnu zakrivljenost puta. 

    ![Calibration Undistorted](https://github.com/misojca/Lane-Lines-Detection/blob/main/result_files/calibration_undistorted.jpg?raw=true)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

    Put ima različite uslove (senke, svetli asfalt, zute i bele linije) pa je trebalo naci nacin
    kako prepoznati lepo linije.

    Koraci:

    Sobel operator: Izracunao sam gradijent po x-osi. Ovo je kljucno jer linije traka teze da budu vertikalne, a Sobel x naglasava nagle promene intenziteta u horizontalnom pravcu.

    B kanal iz LAB prostora dobro detektuje zutu boju linije na putu. Postavljanjem praga za L kanal na > 210 izvodjio sam bele linije.

    Finalna binarna slika dobijena je kao rezultat piksela koji su prepoznati kao zuti zbog B kanala, belih piksela zbog L kanala i 
    piksela koje su oznacene kao ivice koristeci Sobel operator
    
    ![Original](https://github.com/misojca/Lane-Lines-Detection/blob/main/test_images/test3.jpg?raw=true)

    ![Binary Threshold](https://github.com/misojca/Lane-Lines-Detection/blob/main/result_files/binary_thresholded.jpg?raw=true)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

    U originalnoj slici leva i desna linija kolovozne trake se suzavaju. U tom prostoru ne mozemo meriti sirinu trake ili radijus. 

    Koraci:

    Izvorne tacke: Izabrao sam 4 tacke na slici koje u prirodi cine pravougaonik ispred vozila.

    Destinacione tacke: Odredio sam gde te tacke treba da se preslikaju na novoj slici. Cilj je da se trapez ispravi u pravougaonik, cime linije traka postaju paralelne.

    Izracunavanje matrice transformacije: Koristio sam funkciju cv.getPerspectiveTransform(src, dst) da dobijem matricu M. Takodje, izracunao sam i inverznu matricu Minv (cv.getPerspectiveTransform(dst, src)), koja je kljucna za kasnije vraćanje detektovane trake nazad u originalnu perspektivu vozaca

    Primena: Konacna transformacija se vrši funkcijom cv.warpPerspective, koja primenjuje matricu na binarnu sliku.

    Rezultat: Dobijena je slika iz pticje perspektive gde su linije paralelne (ako je put prav) sto olaksava fitovanje polinoma.
    
    
    ![Perspective Transform](https://github.com/misojca/Lane-Lines-Detection/blob/main/result_files/perspective_warped.jpg?raw=true)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
    Nakon sto smo dobili sliku iz pticje perspektive, bele i zute trake bi trebalo da izgledaju kao dve jasne linije, ali zbog suma, senki ili ostecenja na putu, algoritam ne zna koji beli pikseli pripadaju levoj, a koji desnoj traci. Trebalo je osmisliti nacin da ih razdvojimo i matematicki opisemo njihovu putanju.

    Histogram: Posto su trake na dnu slike najblize vozilu i obicno najjasnije, uzeo sam donju polovinu slike i sabrao sve bele piksele po kolonama. Dobijeni histogram ima dva jasna vrha (pika) koji nam govore tacno gde na x-osi pocinju leva i desna linija. To su nase polazne tacke.

    Sliku sam podelio na 9 horizontalnih nivoa. Za svaki nivo sam postavio dva pravougaonika (prozora) odredjene sirine (margin).

    Unutar svakog prozora algoritam trazi sve nenulte (bele) piksele.

    Ukoliko broj pronadjenih piksela prelazi granicu od 50 (minpix), sledeci prozor se ne postavlja direktno iznad prethodnog, vec se pomera levo ili desno tako da mu centar bude u sredini mase pronadjenih piksela.

    Ovo omogucava prozorima da "prate" krivinu puta cak i kada ona naglo skrece.

    Nakon sto su prozori prosli od dna do vrha slike, sakupio sam sve koordinate piksela koji su "upali" u leve prozore i sve koji su upali u desne.
    Koristio sam funkciju np.polyfit(ly, lx, 2) da kroz te tacke provucem najprecizniju mogucu krivu drugog reda.

    Ovaj pristup nam omogucava da dobijemo kontinuiranu liniju cak i tamo gde je isprekidana traka.

    ![Perspective Transform](https://github.com/misojca/Lane-Lines-Detection/blob/main/result_files/lane_pixels_fitted.jpg?raw=true)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

    Izazov u ovom koraku je cinjenica da algoritam posmatra sliku kroz piksele, a nama su potrebni metri za voznju.
    /////////////////////// DODATI FORMULU
    Radijus krivine: Da bih dobio realan radijus u metrima, uradio sam sledece:
    -Definisao sam faktore konverzije na osnovu standarda sirine trake (3.7 metara) i duzine vidljivog dela puta (30 metara).
    -Piksele traka sam pomnozio ovim faktorima i ponovo izracunao koeficijente polinoma (left_fit_cr i right_fit_cr), ali sada u metrima.

    Pozicija vozila u traci (Vehicle Offset): Ovde sam posao od pretpostavke da je kamera montirana tacno na centar vozila.
    -Izracunao sam x koordinate leve i desne linije na samom dnu slike, a zatim pronasao njihovu sredinu.
    -Sredina same slike predstavlja gde se nalazi centar automobila.
    -Odstupanje (Offset): Razlika izmedju centra slike i centra trake nam daje informaciju koliko auto "bezi" ulevo ili udesno. Taj rezultat sam pomnozio   sa xm_per_pix da bih dobio tacno odstupanje u metrima (npr. 0.25m udesno).

    Ove informacije se ispisuju u realnom vremenu na video snimku. Ako je radijus jako veliki (npr. preko 2000m), to nam govori da je put prakticno prav. Offset nam omogucava da razumemo koliko je vozac precizan u drzanju sredine trake.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

    Napravio sam praznu crnu sliku istih dimenzija kao warped binarna slika. Koristeci izracunate tacke polinoma (left_fitx i right_fitx), ispunio sam prostor izmedju njih zelenom bojom pomocu funkcije cv.fillPoly.

    Zelenu masku, koja je u pogledu odozgo, vratio sam u originalnu perspektivu kamere koristeci matricu Minv koju smo sacuvali prilikom transformacije u jednom od prethodnih koraka.
    newwarp = cv.warpPerspective(color_warp, Minv, (w, h))

    Koristeci cv.addWeighted, iskombinovao sam originalnu sliku sa zelenom maskom trake. Postavio sam providnost tako da put ostane jasno vidljiv ispod zelene povrsine.
    result = cv.addWeighted(undist, 1, newwarp, 0.3, 0)

    Na kraju pomocu cv.putText ispisao sam izracunate vrednosti radijusa krivine i odstupanja vozila od centra trake.

    Finalna slika prikazuje jasno osencenu zelenu povrsinu koja precizno prati kolovoznu traku.

    ![Final Result](./result_files/https://github.com/misojca/Lane-Lines-Detection/blob/main/result_files/final_result_image.jpg?raw=true)

### Pipeline (video)
#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

    Konacan snimak:  /result_files/final_video_opencv.avi

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

    Drvece pored puta baca senke koje Sobel operator moze pogresno prepoznati kao liniju. To smo resili oslanjanjem na boju (LAB).

    Ako bi ispred nas bio auto koji potpuno zaklanja linije ili ako bi put bio prekriven snegom ili blatom. U veoma ostrim krivinama bi lose prepoznao levu i desnu traku

    Implementacija algoritma bi bila robusnija ako bi automatski pomerao "source" tacke u zavisnosti od nagiba puta ili brzine kretanja. Potrebno je prosirti algoritam da radi dobro za ulazne video snimke sa vise suma

