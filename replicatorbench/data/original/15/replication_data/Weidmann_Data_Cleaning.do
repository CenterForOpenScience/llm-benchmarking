********************************************************************************
*		WEIDMANN & CALLEN (2013) REPLICATION 
* 		DARPA SCORE PROJECT: 
*		"Replication of a Research Claim from Weidmann & Callen (2013)"
*		Eric L. Sevigny & Jared Greathouse
* 		Contact: esevigny@gsu.edu
********************************************************************************

********************************************************************************
** DATA SOURCES
********************************************************************************

/* Data for this replication come from several sources. Before cleaning and
	merging these data, users will need to obtain the original raw data and
	store these files in a project folder and respective data subfolders on 
	their computer. We provide links to these raw data in this file, and will 
	also post the final dataset to OSF. To ensure data management code runs 
	smoothly once all data has been obtained, raw data should be stored in 
	project subfolders following the naming conventions used below. All data is 
	freely available, although certain data	must be specifically requested. If
	all raw data is obtained and located in proper folders, running this code
	from beginning to end will produce the final analytic dataset used for this
	replication. Again, we will post this final analytic dataset to OSF, and
	provide direct links or other sourcing information for all raw data. */
	
*** ELECTION DATA
/* 2014 Afghanistan election data from the National Democratic Institute (NDI)
	is freely available from the following links. Store locally in 
	"Election_Data" project subfolder.

	Afghanistan 2014 Presidential election data and information:
		https://afghanistanelectiondata.org/about/2014/data-sources
	
	Direct downloads of April 5, 2014 election data:
		https://cdn.jsdelivr.net/gh/developmentseed/af-elections-data@master/2014-presidential-firstround/downloads/2014_Results_First.csv
		https://cdn.jsdelivr.net/gh/developmentseed/af-elections-data@master/2014-presidential-firstround/downloads/2014_PollingCenters_First.csv
		
	Direct download of June 14, 2014 election data:
		https://cdn.jsdelivr.net/gh/developmentseed/af-elections-data@master/2014-presidential-runoff/downloads/2014_Results_Runoff.csv
		https://cdn.jsdelivr.net/gh/developmentseed/af-elections-data@master/2014-presidential-runoff/downloads/2014_PollingCenters_Runoff.csv
		
	Direct download of province and district data:
		https://cdn.jsdelivr.net/gh/developmentseed/af-elections-data@master/2014-presidential-runoff/downloads/2014_rosetta.csv */

*** VIOLENCE DATA: SIGACTs 
/* Afghanistan SIGACTs "Significant Activity" data on violence in Afghanistan
	originally obtained by Vincent Bauer from US CENTCOM is available from the 
	following links. Store locally in "Violence_Data" project subfolder.
	
	Bauer's data page:
		https://stanford.edu/~vbauer/data.html
		
	Direct download of data is available here:
		https://stanford.edu/~vbauer/files/data/sigacts/AfgSigacts.xlsx */

*** DEVELOPMENT DATA
/* Development measures come from the Afghanistan Living Conditions Survey 
	(ALCS) 2013-14. The ALCS is a regular social survey administered by Afghan 
	government. The data can be requested from the National Statistic and 
	Information Authority (NSIA) data office: https://nsia.gov.af/home;
	<data.sharing@nsia.gov.af>. These data were provided freely by NSIA, but a
	data use agreement precludes sharing the raw data. The specific data used 
	for this replication were provided in the following Stata data files: 
	'District Code.dta' and	'H_04-09.dta'. Store locally in "Development_Data" 
	project subfolder. */

*** GEOGRAPHIC DATA
/* The World Bank provides district-level Afghanistan data at the following link:
		https://www.worldbank.org/en/data/interactive/2019/08/01/afghanistan-district-level-visualization	

	To download elevation data, users must select the indicator: 'Geography - 
		Mean Elevation' and follow the download instructions. Store locally in 
		"Geographic_Data" project subfolder.*/

*** POPULATION DATA
/* 2004-2020 Population data from Afghanistan's National Statistics and 
	Information Authority (NSIA) (formerly the Central Statistics Office) 
	has been assembled by Colin Cookman and is downloadable from github here:
		https://github.com/colincookman/afghanistan_district_population_data/blob/master/cso_district_population_estimates_2004-2020.csv
	Store locally in "Population_Data" project subfolder. */

********************************************************************************
** DATA CLEANING 
********************************************************************************

/// Users should insert project directory between quotes before running code:
cap cd "INSERT PROJECT DIRECTORY HERE" 

/// Stata MP version
version 16.1

*** INSTALL COMMANDS
net install xcollapse, from(http://fmwww.bc.edu/RePEc/bocode/x)
net install tab_chi, from(http://fmwww.bc.edu/RePEc/bocode/t)
net install egenmore, from(http://fmwww.bc.edu/RePEc/bocode/e)
net install geodist, from(http://fmwww.bc.edu/RePEc/bocode/g)

*** ELECTION DATA
/// Clean and merge datasets for first (April 5, 2014) election

* Import and clean polling center data (N=6,775 polling centers)
import delimited "Election_Data\2014_PollingCenters_First.csv", clear

* Extract district ID from polling center ID
gen distid = floor(pc_code/1000)

* Trim extra spaces
gen provincec=trim(province)
gen districtc=trim(district)

* Kandahar province recorded in Pashto; replace with English
replace provincec="Kandahar" if iec_prov_id==27

* Replace recorded subdistrict names with top-level district name
replace districtc="Kabul" if distid==101 
replace districtc="Herat" if distid==3201 

save "Election_Data\2014_PollingCenters_First.dta", replace
clear

* Import and clean first round election data (N=18,824 polling stations)
import delimited "Election_Data\2014_Results_First.csv" 

gen provincee=proper(province)
gen districte=proper(district)
rename total votes // apply meaningful variable name

* Extract district ID from polling center ID
gen distid = floor(pc_number/1000)	

/* Replace missing/incorrect province and district name based on district ID; 
	cross-referenced with https://arcg.is/PDDfz District Lookup Tool. */
replace provincee="Ghor" if distid==2310
replace districte="Saghar" if distid==2310
replace districte="Darah Suf-E-Bala" if distid==2007

* Generate last digit variable from vote total 
gen lastdig=mod(votes, 10)

* Perform last digit test by district and create fraud measure at p<0.05 cut
gen pval=.
levelsof distid, local(levels)
foreach l of local levels {
	    di `l'
		capture noisily: chitest lastdig if distid==`l', count sep(0)
		replace pval=r(p) if distid==`l'
}
recode pval (.05/max=0) (.=.) (else=1), gen(fraud)

/* Note: 4 districts had one polling center each, and thus the last digit test
	could not be run with n = 1. We assumed no evidence of fraud in these 
	districts. One district with 11 polling centers all reported 600 votes each; 
	however, last digit test cannot run with no variation in test of uniform 
	distribution. This vote pattern is considered highly irregular by election 
	observers, so we coded this district as showing evidence of fraud. */
replace fraud = 0 if inlist(distid, 1112, 1118, 2608, 3308)
replace fraud = 1 if distid==3107

* Rename polling center ID to be consistent across datasets
rename pc_number pc_code

* Identify as first 2014 election 
gen elect=1

* Create indicator for open polling stations
gen psno=1

* Aggregate polling station level data to polling center level
xcollapse (first) provincee districte distid (mean) fraud elect ///
	(sum) votes psno, by(pc_code) norestore

* Merge polling center data
merge 1:1 pc_code using "Election_Data\2014_PollingCenters_First.dta"
drop _merge

* Create indicator for open polling centers
recode votes (0/max=1) (.=0), gen(pcno)

* Create indicator for total polling centers
gen pcn=1

* Create indicator for total polling stations
clonevar psn=ps_total

* Aggregate polling center level election data to district level
xcollapse (first) provincee districte provincec districtc ///
	(mean) fraud elect lat lon (sum) votes pcn pcno psn psno, by(distid) ///
	norestore

order provincee districte provincec districtc distid lat lon fraud elect ///
	votes pcn pcno psn psno

* Create polling center/station variables
gen pcnx=pcn-pcno
gen psnx=psn-psno
gen pcx=100*(pcnx/pcn)
gen psx=100*(psnx/psn)
format pcx psx %6.1fc

* Label variables
lab var provincee "Province (elect)"
lab var districte "District (elect)"
lab var provincec "Province (poll)"
lab var districtc "District (poll)"
lab var distid "District ID"
lab var lat "Latitude"
lab var lon "Longitude"
lab var fraud "Election Fraud"
lab var elect "Election Cycle"
lab var votes "# Votes"
lab var pcn "# Planned Polling Centers"
lab var pcno "# Open Polling Centers"
lab var pcnx "# Closed Polling Centers"
lab var pcx "% Polling Centers Closed"
lab var psn "# Planned Polling Stations"
lab var psno "# Open Polling Stations"
lab var psnx "# Closed Polling Stations"
lab var psx "% Polling Stations Closed"

save "Election_Data\2014_Results_First.dta", replace
clear

* Import and clean election district information (N=407)
import delimited "Election_Data\2014_rosetta.csv" 
rename (agcho_prov agcho_dist iec_id_fix) (province district distid)
replace district=subinstr(district, "- ", "-", .)
replace district=subinstr(district, "  ", " ", .)
replace district="Kiti" if district=="kiti"

/* Replace missing/incorrect district IDs based on district; 
	cross-referenced with https://arcg.is/PDDfz District Lookup Tool. */
replace distid=1119 if district=="Nawa"
replace distid=2116 if district=="Sharak-e-Hayratan"
replace distid=2611 if district=="Kakar"
replace distid=3012 if district=="Baghran"
replace distid=3013 if district=="Deh shu"
replace distid=2007 if district=="Darah Suf-e-Bala"

keep province district distid 

* Add one district that appears in other election datasets, increasing to N=408
set obs `=_N+1'
replace province="Laghman" if province==""
replace district="Badpakh" if district==""
replace distid=706 if distid==.

lab var province "Province"
lab var district "District"
lab var distid "District ID"

sort distid

save "Election_Data\2014_rosetta.dta", replace

* Merge election results 
merge 1:1 distid using "Election_Data\2014_Results_First.dta"
drop _merge

* Drop data-checking jursdiction variables
/* Before dropping the following variables, we visually inspected the quality of
	the district-level match by entering the following command: 
		browse province* district* distid */ 
drop provincee districte provincec districtc

save "Election_Data\2014_First Round Election Results.dta", replace
clear

/// Clean and merge datasets for runoff (June 14, 2014) election

* Import and clean polling center data (N=6,365 polling centers)
clear
import delimited "Election_Data\2014_PollingCenters_Runoff.csv"

* Extract district ID from polling center ID
gen distid = floor(pcnumber/1000)	

* Trim extra spaces
gen provincec=trim(province)
gen districtc=trim(district)

* Kandahar province recorded in Pashto; replace with English
replace provincec="Kandahar" if iec_prov_id==27

* Replace recorded subdistrict names with full district name
replace districtc="Kabul" if distid==101 
replace districtc="Herat" if distid==3201 

* Rename polling center ID consistently across datasets
rename pcnumber pc_code

* Clean longitude variable name
rename v1 lon

save "Election_Data\2014_PollingCenters_Runoff.dta", replace
clear

* Import and clean first round election data (N=22,469 polling stations)
import delimited "Election_Data\2014_Results_Runoff.csv"

gen provincee=proper(province)
gen districte=proper(district)
rename total votes // apply meaningful variable name

* Extract district ID from polling center ID
gen distid = floor(pc_number/1000)	

* Generate last digit variable from vote total 
gen lastdig=mod(votes, 10)

* Perform last digit test by district and create fraud measure at p<0.05 cut
gen pval=.
levelsof distid, local(levels)
foreach l of local levels {
	    di `l'
		capture noisily: chitest lastdig if distid==`l', count sep(0)
		replace pval=r(p) if distid==`l'
}
recode pval (.05/max=0) (.=.) (else=1), gen(fraud)

/* Note: 2 districts had one polling center, and thus the last digit test
	could not be run with n = 1. We assumed no evidence of fraud in these
	districts. */
replace fraud = 0 if inlist(distid, 1112, 2403)

* Identify as runoff 2014 election 
gen elect=2

* Create indicator for open polling stations
gen psno=1

* Rename polling center ID consistently across datasets
rename pc_number pc_code

* Aggregate polling station level election data to polling center level
xcollapse (first) provincee districte distid (mean) fraud elect ///
	(sum) votes psno, by(pc_code) norestore

* Merge polling center data
merge 1:1 pc_code using "Election_Data\2014_PollingCenters_Runoff.dta"
drop _merge

* Create indicator for open polling centers
recode votes (0/max=1) (.=0), gen(pcno)

* Create indicator for total polling centers
gen pcn=1

* Create indicator for total polling stations
clonevar psn=ps_total

* Aggregate polling center level election data to district level
xcollapse (first) provincee districte provincec districtc ///
	(mean) fraud elect lat lon (sum) votes pcn pcno psn psno, by(distid) ///
	norestore

order provincee districte provincec districtc distid lat lon fraud elect ///
	votes pcn pcno psn psno
	
* Create polling center/station variables
gen pcnx=pcn-pcno
gen psnx=psn-psno
gen pcx=100*(pcnx/pcn)
gen psx=100*(psnx/psn)
format pcx psx %6.1fc
	
* Label variables
lab var provincee "Province (elect)"
lab var districte "District (elect)"
lab var provincec "Province (poll)"
lab var districtc "District (poll)"
lab var distid "District ID"
lab var lat "Latitude"
lab var lon "Longitude"
lab var fraud "Election Fraud"
lab var elect "Election Cycle"
lab var votes "# Votes"
lab var pcn "# Planned Polling Centers"
lab var pcno "# Open Polling Centers"
lab var pcnx "# Closed Polling Centers"
lab var pcx "% Polling Centers Closed"
lab var psn "# Planned Polling Stations"
lab var psno "# Open Polling Stations"
lab var psnx "# Closed Polling Stations"
lab var psx "% Polling Stations Closed"

save "Election_Data\2014_Results_Runoff.dta", replace
clear
	
** Merge election results 
use "Election_Data\2014_rosetta.dta"
merge 1:1 distid using "Election_Data\2014_Results_Runoff.dta"
drop _merge

* Drop data-checking jursdiction variables
/* Before dropping the following variables, we visually inspected the quality of
	the district-level match by entering the following command: 
		browse province* district* distid */ 
drop provincee districte provincec districtc

save "Election_Data\2014_Runoff Election Results.dta", replace
clear

*** VIOLENCE DATA
clear
import excel "Violence_Data\AfgSigacts.xlsx", sheet("AfgSigactsOutFin") ///
	firstrow case(lower) // N=431,547 | BE PATIENT DURING IMPORT...

* Remove alphamumeric values from latitude and longitude variables
replace ddlat = substr(ddlat, 1, length(ddlat) - 1) ///
	if substr(ddlat, -1, 1) ==  "N"
replace ddlat = substr(ddlat, 1, length(ddlat) - 1) ///
	if substr(ddlat, -1, 1) ==  "S"
destring ddlat, replace
replace ddlon = substr(ddlon, 1, length(ddlon) - 1) ///
	if substr(ddlon, -1, 1) ==  "E"
replace ddlon = substr(ddlon, 1, length(ddlon) - 1) ///
	if substr(ddlon, -1, 1) ==  "W"
destring ddlon, replace

rename (primaryeventtype primaryeventcategroy dateoccurred) ///
	(event_typ event_cat date)

* Retain events that occurred in the 2014 election year
keep if year(date)==2014 // N=76,260 observations retained

/* Retain events that match violence definition. This includes enemy 
	actions involving (i) assassinations, direct fire, or small arms fire, or 
	(ii) IED/mine explosive hazards. */
keep if (event_typ=="Enemy Action" & inlist(event_cat, ///
	"Assassination", "Direct Fire", "IED Found and Cleared", "SAFIRE")) | ///
	(event_typ=="Explosive Hazard" & (strpos(event_cat, "IED") | ///
	strpos(event_cat, "Mine"))) // N=13,550 observations retained

* Extract district name from event description and clean 
egen loc1=ends(title), punct(RPT) tail trim
egen loc2=sieve(loc1), keep(a o s)
gen districtv=trim(loc2)
gen pos=strpos(districtv, "'")
replace district=regexr(districtv,"'","") if pos==1
drop loc1 loc2 pos
replace districtv="Chahar Burjak" if districtv=="Chahar Burjak RD KDK" // 1 change
replace districtv="Sangin" if districtv=="Sangin ANA #" // 1 change
replace districtv="Washer" if districtv=="Washer st KDK" // 1 change
replace districtv="Terayzai" if districtv=="Terayzai ('Ali Sher)" // 1 change
replace districtv="Manduzai" if districtv=="Manduzai (Isma il Khel)" // 3 changes
replace districtv="Musa Khel" if districtv=="Musa Khel (Mangal)" // 2 changes
replace districtv="Sabari" if districtv=="Sabari (Ya qubi)" // 2 changes

* Drop events located outside Afghanistan or of unknown location
drop if inlist(districtv, "BAJAUR", "KHYBER", "KURRAM", ///
	"SOUTH WAZIRISTAN", "") // 42 observations deleted

/* Because these data contain no unique ID for districts, we manually create one 
	to facilitate accurate dataset merges. We investigated fuzzy matching 
	routines (e.g., reclink2), but these performed unsatsfactorily in this 
	context. The following code creates a unique district ID consistent with the 
	Election datasets for subsequent merge. */

#delimit ;

local id 101 102 103 104 105 106 107 108 109 111 113 114 201 202 204 205 206 207 301 302 303 304 305 306 307 308 310 401 402 403 404 405 406 408 501 502 503 504 505 506 507 601 602 603 604 605 606 607 608 609 610 611 613 614 615 616 617 618 619 620 621 622 701 702 703 704 705 706 806 901 902 903 904 905 906 907 910 911 912 1101 1102 1103 1104 1105 1106 1107 1108 1109 1110 1111 1112 1113 1114 1115 1117 1118 1119 1201 1202 1203 1204 1205 1206 1207 1208 1209 1210 1211 1212 1213 1214 1215 1216 1217 1218 1219 1301 1302 1303 1304 1305 1306 1307 1308 1309 1310 1311 1312 1313 1401 1402 1403 1404 1405 1406 1407 1408 1409 1410 1411 1412 1413 1501 1502 1503 1504 1505 1506 1507 1508 1509 1510 1511 1512 1513 1514 1515 1602 1603 1604 1605 1606 1608 1701 1702 1703 1704 1706 1708 1710 1712 1715 1716 1717 1718 1726 1728 1801 1803 1804 1805 1807 1808 1809 1810 1811 1812 1814 1815 1816 1817 1901 1902 1903 1904 1905 1906 1907 2001 2002 2003 2006 2101 2102 2103 2106 2107 2108 2109 2110 2111 2112 2113 2115 2116 2201 2202 2203 2204 2205 2206 2207 2301 2302 2303 2304 2305 2306 2308 2309 2310 2402 2403 2405 2409 2501 2502 2503 2504 2505 2506 2601 2602 2603 2604 2605 2606 2607 2608 2609 2610 2611 2701 2702 2703 2704 2705 2706 2707 2708 2709 2710 2711 2712 2713 2714 2715 2716 2717 2801 2802 2803 2804 2805 2806 2807 2808 2809 2810 2811 2901 2902 2903 2904 2905 2906 2907 2908 2909 2910 2911 2912 2913 2914 3001 3002 3003 3004 3005 3006 3007 3008 3009 3010 3011 3013 3014 3101 3102 3103 3104 3105 3106 3107 3201 3202 3203 3204 3205 3206 3207 3208 3209 3210 3211 3212 3213 3214 3215 3216 3301 3302 3303 3304 3305 3306 3307 3308 3310 3311 3401 3402 3403 3404 3405 3406  ;

local dist ""Kabul" "Paghman" "Chahar Asyab" "Bagrami" "Deh-e Sabz" "Shakar Darah" "Musahi" "Mir Bachah Kot" "Khak-e Jabar" "Gul Darah" "Istalif" "Qarah Bagh" "Mahmud-e Raqi" "Hisah-e Dowum-e Kohistan" "Hisah-e Awal-e Kohistan" "Nejrab" "Tagab" "Alah Say" "Charikar" "Bagram" "Shinwari" "Sayyid Khayl" "Jabal us Saraj" "Salang" "Siahgird (Ghorband)" "Koh-e Safi" "Shaykh 'Ali" "Maidan Shahr" "Nerkh" "Jalrayz" "Chak-e Wardak" "Sayyidabad" "Daymirdad" "Jaghatu" "Pul-e 'Alam" "Baraki Barak" "Charkh" "Khoshi" "Muhammad Aghah" "Kharwar" "Azrah" "Jalalabad" "Behsud" "Surkh Rod" "Chaparhar" "Kamah" "Kuz Kunar" "Rodat" "Khugyani" "Bati Kot" "Deh Bala" "Pachir wa Agam" "Kot" "Goshtah" "Achin" "Shinwar" "Mohmand Darah" "La'lpur" "Sherzad" "Nazyan" "Hisarak" "Dur Baba" "Mehtar Lam" "Qarghah'i" "Alisheng" "Alingar" "Dowlat Shah" "Bad Pech" "Shutul" "Pul-e Khumri" "Dahanah-ye Ghori" "Doshi" "Nahrin" "Baghlan-e Jadid" "Khinjan" "Andarab" "Burkah" "Talah wa Barfak" "Pul-e Hisar" "Ghazni" "Wali Muhammad Shahid Khugyani" "Khwajah Omari" "Waghaz" "Deh Yak" "Bahram-e Shahid (Jaghatu)" "Andar" "Zanakhan" "Rashidan" "Nawur" "Qarah Bagh" "Giro" "Ab Band" "Jaghuri" "Muqer" "Gelan" "Ajristan" "Nawah" "Sharan" "Mota Khan" "Yosuf Khel" "Yahya Khel" "Sar Rowzah" "Omnah" "Zarghun Shahr" "Gomal" "Jani Khel" "Sarobi" "Orgun" "Ziruk" "Nikeh" "Bermal" "Giyan" "Dilah wa Khoshamand" "Wazah Khwah" "Wur Mamay" "Terwo" "Gardez" "Ahmadabad" "Zurmat" "Shwak" "Dzadran" "Sayyid Karam" "Jaji" "Lajah-Ahmad Khel" "Jani Khel" "Tsamkani" "Dand-Patan" "Lajah Mangal" "Mirzakah" "Khost (Matun)" "Manduzai" "Gurbuz" "Tanai" "Musa Khel" "Nadir Shah Kot" "Sabari" "Terayzai" "Bak" "Qalandar" "Sperah" "Shamul (Dzadran)" "Jaji Maidan" "Asadabad" "Marawarah" "Watahpur" "Narang" "Sar Kani" "Shigal wa Sheltan" "Darah-ye Pech" "Bar Kunar (Asmar)" "Tsowkey" "Khas Kunar" "Ghaziabad" "Dangam" "Chapah Darah" "Nurgal" "Nari" "Waygal" "Wama" "Nurgaram" "Do Ab" "Kamdesh" "Barg-e Matal" "Faizabad" "Argo" "Arghanj Khwah" "Yaftal-e Sufla" "Baharak" "Kohistan" "Jurm" "Shuhada" "Kishim" "Warduj" "Tagab" "Yamgan" "Kiran wa Munjan" "Wakhan" "Taloqan" "Baharak" "Bangi" "Chal" "Kalafgan" "Farkhar" "Khwajah Ghar" "Rustaq" "Ishkamish" "Dasht-e Qal'ah" "Khwajah Bahawuddin" "Darqad" "Chah Ab" "Yangi Qal'ah" "Kunduz" "Chahar Darah" "Aliabad" "Khanabad" "Imam Sahib" "Archi" "Qal'ah-ye Zal" "Aibak" "Hazrat-e Sultan" "Khuram wa Sar Bagh" "Dara-ye Suf-e Pa'in" "Mazar-e Sharif" "Nahr-e Shahi" "Dehdadi" "Balkh" "Sholgarah" "Chimtal" "Dowlatabad" "Khulm" "Chahar Bolak" "Shor Tepah" "Kaldar" "Zari" "Shahrak-e Hairatan" "Sar-e Pul" "Sayad" "Kohistanat" "Sozmah Qal'ah" "Sangcharak" "Gosfandi" "Balkhab" "Chaghcharan" "Do Lainah" "Dowlatyar" "Chahar Sadah" "Pasaband" "Shahrak" "Taywarah" "Tulak" "Saghar" "Shahristan" "Gizab" "Khedir" "Kajran" "Tarin Kot" "Deh Rawud" "Chorah" "Shahid-e Hasas" "Khas Uruzgan" "Chinartu" "Qalat" "Tamek wa Jaldak" "Shinkai" "Mizan" "Arghandab" "Shah Joy" "Daychopan" "Atghar" "Now Bahar" "Shamulzai" "Khak-e Afghan" "Kandahar" "Arghandab" "Daman" "Panjwa'i" "Zharay" "Shah Wali Kot" "Khakrez" "Arghistan" "Ghorak" "Maiwand" "Spin Boldak" "Nesh" "Mya Neshin" "Shorabak" "Ma'ruf" "Registan" "Dand" "Shibirghan" "Khwajah Do Koh" "Khanaqa" "Mingajik" "Qush Tepah" "Khamyab" "Aqchah" "Faizabad" "Mardian" "Qarqin" "Darzab" "Maimanah" "Pashtun Kot" "Khwajah Sabz Posh" "Almar" "Bal Chiragh" "Shirin Tagab" "Qaisar" "Gurziwan" "Dowlatabad" "Kohistan" "Qaram Qol" "Qurghan" "Andkhoy" "Khan-e Chahar Bagh" "Lashkar Gah" "Nad 'Ali" "Nawah-ye Barakzai" "Nahr-e Saraj" "Washer" "Garm Ser" "Now Zad" "Sangin" "Musa Qal'ah" "Kajaki" "Reg-e Khan Neshin" "Dishu" "Marjah" "Qal'ah-ye Now" "Ab-e Kamari" "Muqur" "Qadis" "Murghab" "Jawand" "Ghormach" "Herat" "Injil" "Nizam-e Shahid (Guzarah)" "Karukh" "Zindah Jan" "Pashtun Zarghun" "Kushk (Rabat-e Sangi)" "Gulran" "Adraskan" "Kushk-e-Kohnah" "Ghorian" "Obeh" "Kohsan" "Shindand" "Farsi" "Chisht-e-Sharif" "Farah" "Pusht-e Rod" "Khak-e-Safayd" "Qal'ah-ye Kah" "Shayb Koh" "Bala Boluk" "Anar Darah" "Bakwah" "Gulistan" "Pur Chaman" "Zaranj" "Kang" "Chakhansur" "Chahar Burjak" "Khash Rod" "Delaram"" ;

#delimit cr

cap drop distid 
gen distid=. 

loc n: word count `dist'
forv i = 1/`n' {
	loc a: word `i' of `id'
	loc b: word `i' of `dist' 
    qui: replace distid= `a' if districtv== "`b'"
}
	
/* Some districts from different provinces have the same name. Because the 
	Sigacts	data do not report province name, we distinquish these districts by 
	using reported latitude/longitude coordinates. */
replace distid=2605 if districtv=="Arghandab" & ddlat>32	
replace distid=1706 if districtv=="Baharak" & ddlon>70.5
replace distid=2109 if districtv=="Dowlatabad" & ddlon>66.5
replace distid=1701 if districtv=="Faizabad" & ddlon>70
replace distid=1209 if districtv=="Jani Khel" & ddlon>69
replace distid=1708 if districtv=="Kohistan" & ddlon>66
replace distid=114 if districtv=="Qarah Bagh" & ddlon>69
replace distid=115 if districtv=="Sarobi" & ddlat>34
replace distid=206 if districtv=="Tagab" & (ddlat>34.85 & ddlon<69.66)

/* Generate four variables that capture violent SIGACT events both (i) 5 or 60 
	dates around the (ii) first (4/5/14) and second (6/14/14) elections. The five-
	day window included election day and the following four days; the sixty-day 
	window includes the 60 days preceding election day. */
gen sigact_5_1=inrange(date, td(05apr2014), td(09apr2014))
gen sigact_60_1=inrange(date, td(03feb2014), td(04apr2014))
gen sigact_5_2=inrange(date, td(14jun2014), td(18jun2014))
gen sigact_60_2=inrange(date, td(14apr2014), td(13jun2014))

* Aggregate number of violent events to district level
xcollapse (first) districtv (sum) sigact*, by(distid) norestore 

* Label variables
lab var distid "District ID"
lab var districtv "District"
lab var sigact_5_1 "Violent Events (5 days, 1st vote)"
lab var sigact_60_1 "Violent Events (60 days, 1st vote)"
lab var sigact_5_2 "Violent Events (5 days, 2nd vote)"
lab var sigact_60_2 "Violent Events (60 days, 2nd vote)"

save "Violence_Data\AfgSigacts_2014_district.dta", replace
clear

*** DEVELOPMENT DATA
* Load and clean district information
clear
use "Development_Data\District Code.dta"
cap clonevar distid=DistrictCode
save "Development_Data\District Code.dta", replace

* Load and clean development data
clear
use "Development_Data\H_04-09.dta"
gen distid=(q_1_1*100)+q_1_2 // appends province and district components
lab var distid "District ID"
merge m:1 distid using "Development_Data\District Code.dta", ///
	keepusing(ProvinceName DistrictName)
rename (ProvinceName DistrictName) (provinced districtd)
decode q_1_1a, gen(province_alcs)
replace districtd=proper(districtd)
replace provinced=proper(provinced)
replace provinced=province_alcs if provinced==""
drop _merge province_alcs

* Create household electricity measure
egen electric=anymatch(q_4_11_*), v(1)
egen electm=rowmiss(q_4_11_*)
replace electric=. if electm==9
drop electm

* Create household expenditures measure
egen expendf=rowtotal(q_4_15_*), missing // Sum past-month fuel expenditures
egen expendcm=rowtotal(q_8_8_*), missing // Sum past-month commodity expenditures
egen expendcy=rowtotal(q_8_9_*), missing // Sum past-year commodity expenditures
replace expendcy=expendcy/12 // Normalize to monthly
egen expend=rowtotal(expendf expendcm expendcy), missing // Sum all expenditures
drop expendf expendcm expendcy
clonevar tot_expend=expend

* Collapse to district level using household sampling weights
xcollapse (first) provinced districtd (mean) electric (sum) tot_expend ///
	[pw=hh_weight], by(distid) norestore
	
* Label variables
lab var provinced "Province"
lab var districtd "District"	
lab var distid "District ID"
lab var electric "Proportion Households w/ Electricity"
lab var tot_expend "Total Past-Month Houseshold Expenditures"
format electric %11.1fc
format tot_expend %16.0fc

order provinced districtd distid electric tot_expend 

save "Development_Data\ALCS_1314.dta", replace
clear
	
*** GEOGRAPHIC DATA
* Load and clean data
clear
import delimited "Geographic_Data\Mean_Elevation.csv", rowrange(:401) 	
	
rename (ïdistname provname continuous1) (districtg provinceg elevation)
lab var districtg "District"
lab var provinceg "Province"
lab var elevation "Average District Elevation (m)"
replace districtg=proper(districtg)
replace provinceg=proper(provinceg)
destring elevation, replace ignore(",")

order provinceg districtg, first
sort provinceg districtg, stable	

/* Because these data contain no unique ID for districts, we manually create one 
	to facilitate accurate dataset merges. We investigated fuzzy matching 
	routines (e.g., reclink2), but these performed unsatsfactorily in this 
	context. The following code creates a unique district ID consistent with the 
	Election datasets for subsequent merge. */
	
#delimit ;

local id 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 201 202 203 204 205 206 207 301 302 303 304 305 306 307 308 309 310 401 402 403 404 405 406 407 408 409 501 502 503 504 505 506 507 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 701 702 703 704 705 706 801 802 803 804 805 806 807 808 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 1001 1002 1003 1004 1005 1006 1007 1101 1102 1103 1104 1105 1106 1107 1108 1109 1110 1111 1112 1113 1114 1115 1116 1117 1118 1119 1201 1202 1203 1204 1205 1206 1207 1208 1209 1210 1211 1212 1213 1214 1215 1216 1217 1218 1219 1301 1302 1303 1304 1305 1306 1307 1308 1309 1310 1311 1312 1313 1401 1402 1403 1404 1405 1406 1407 1408 1409 1410 1411 1412 1413 1501 1502 1503 1504 1505 1506 1507 1508 1509 1510 1511 1512 1513 1514 1515 1601 1602 1603 1604 1605 1606 1607 1608 1701 1702 1703 1704 1705 1706 1707 1708 1709 1710 1711 1712 1713 1714 1715 1716 1717 1718 1719 1720 1721 1722 1723 1724 1725 1726 1727 1728 1801 1802 1803 1804 1805 1806 1807 1808 1809 1810 1811 1812 1813 1814 1815 1816 1817 1901 1902 1903 1904 1905 1906 1907 2001 2002 2003 2004 2005 2006 2007 2101 2102 2103 2104 2105 2106 2107 2108 2109 2110 2111 2112 2113 2114 2115 2116 2201 2202 2203 2204 2205 2206 2207 2301 2302 2303 2304 2305 2306 2307 2308 2309 2310 2401 2402 2403 2404 2405 2406 2407 2408 2409 2501 2502 2503 2504 2505 2506 2601 2602 2603 2604 2605 2606 2607 2608 2609 2610 2611 2701 2702 2703 2704 2705 2706 2707 2708 2709 2710 2711 2712 2713 2714 2715 2716 2717 2801 2802 2803 2804 2805 2806 2807 2808 2809 2810 2811 2901 2902 2903 2904 2905 2906 2907 2908 2909 2910 2911 2912 2913 2914 3001 3002 3003 3004 3005 3006 3007 3008 3009 3010 3011 3012 3013 3014 3101 3102 3103 3104 3105 3106 3107 3201 3202 3203 3204 3205 3206 3207 3208 3209 3210 3211 3212 3213 3214 3215 3216 3301 3302 3303 3304 3305 3306 3307 3308 3309 3310 3311 3401 3402 3403 3404 3405 3406 2223 ;

local dist ""Kabul" "Paghman" "Chahar Asyab" "Bagrami" "Dehsabz" "Shakar Dara" "Musahi" "Mir Bacha Kot" "Khak-I-Jabar" "Kalakan" "Guldara" "Farza" "Estalef" "Qara Bagh" "Surubi" "Kapisa (Mahmood Raqi)" "Hissa-I-Duwumi Kohistan" "Koh Band" "Hissa-I-Awal Kohistan" "Nijrab" "Tagab" "Alasai" "Parwan (Charikar)" "Bagram" "Shinwari" "Sayyid Khel" "Jabulussaraj" "Salang" "Syahgird (Ghorband)" "Koh-I-Safi" "Surkhi Parsa" "Shaykh Ali" "Wardak (Maidan Shahr)" "Nerkh" "Jalrez" "Chak-I-Wardak" "Sayyidabad" "Daimirdad" "Hissa-I- Awal Behsud" "Jaghatu" "Markaz-I-Behsud" "Logar (Puli Alam)" "Baraki Barak" "Charkh" "Khushi" "Mohammad Agha" "Khar War" "Azra" "Nangarhar (Jalalabad)" "Behsud" "Surkh Rud" "Chaparhar" "Kama" "Kuzkunar" "Rodat" "Khugyani" "Bati Kot" "Deh Bala" "Pachir Wagam" "Darah-I-Noor" "Kot" "Goshta" "Achin" "Shinwar" "Muhmand Dara" "Lalpoor" "Sher Zad" "Nazyan" "Hesarak" "Dur Baba" "Laghman (Mehterlam)" "Qarghayee" "Alishing" "Alingar" "Dawlat Shah" "Bad Pash" "Panjsher (Bazarak)" "Rukha" "Darah" "Hissa-I-Awal (Khinj )" "Unaba" "Shutul" "Paryan" "Abshar" "Baghlan (Pul-I-Khumri)" "Dahana-I-Ghuri" "Dushi" "Nahreen" "Baghlan-I-Jadeed" "Khinjan" "Andarab" "Deh Salah" "Jalga" "Burka" "Tala Wa Barfak" "Pul-I-Hisar" "Khost Wa Firing" "Gozargah-I-Noor" "Firing Wa Gharu" "Bamyan (Bamyan)" "Shebar" "Saighan" "Kahmard" "Yakawlang" "Panjab" "Waras" "Ghazni" "Shahid Khugyani" "Khwaja Omari" "Waghaz" "Deh Yak" "Jaghatu" "Andar" "Zanakhan" "Rashidan" "Nawur" "Qara Bagh" "Giro" "Ab Band" "Jaghuri" "Muqur" "Malistan" "Gelan" "Ajristan" "Nawa" "Paktika (Sharan)" "Mata Khan" "Yosuf Khel" "Yahya Khel" "Sar Rawza" "Omna" "Zarghun Shahr" "Gomal" "Jani Khel" "Surubi" "Urgoon" "Ziruk" "Nika" "Barmal" "Giyan" "Dila Wa Khushamand" "Wazakhwah" "Wor Mamay" "Turwo" "Paktya (Gardez)" "Ahmadaba" "Zurmat" "Shwak" "Wuza Jadran" "Sayyid Karam" "Jaji" "Laja Ahmad Khel" "Jani Khel" "Samkani" "Dand Patan" "Lajah Mangal" "Mirzakah" "Khost" "Manduzay (Esmayel Khel)" "Gurbuz" "Tanay" "Musa Khel" "Nadirshah Kot" "Sabari (Yaqubi)" "Ali Sher (Terezayi)" "Baak" "Qalandar" "Spera" "Shamul" "Jaji Maidan" "Kunarha (Asad Abad)" "Mara Wara" "Watapoor" "Narang Wa Badil" "Sar Kani" "Shigal Wa Sheltan" "Dara-I-Pech" "Bar Kunar" "Sawkai (Chawkay)" "Khas Kunar" "Ghazi Abad" "Dangam" "Chapa Dara" "Noorgal" "Nari" "Nooristan (Paroon)" "Waygal" "Wama" "Noor Gram" "Duab" "Kamdesh" "Mandol" "Bargi Matal" "Badakhshan (Faiz Abad)" "Argo" "Arghanj Khwah" "Yaftal -I-Sufla" "Khash" "Baharak" "Darayim" "Kohistan" "Yawan" "Jurm" "Tashkan" "Shuhada" "Shahri Buzurg" "Raghistan" "Kishm" "Wardooj" "Tagab" "Yamgan" "Shighnan" "Khwahan" "Kufab" "Darwaz-I- Payin (Mamay)" "Eshkashim" "Shiki" "Zebak" "Kiran Wa Menjan" "Darwaz-I- Bala (Nesay)" "Wakhan" "Takhar (Taluqan)" "Hazar Sumuch" "Baharak" "Bangi" "Chal" "Namak Ab" "Kalafgan" "Farkhar" "Khwaja Ghar" "Rustaq" "Eshkamesh" "Dashti Qala" "Warsaj" "Khwaja Bahawuddin" "Darqad" "Chahab" "Yangi Qala" "Kunduz" "Char Darah" "Ali Abad" "Khan Abad" "Hazrati Imam Sahib" "Dasht-I-Archi" "Qala-I-Zal" "Samangan (Aybak)" "Hazrat-I-Sultan" "Khuram Wa Sarbagh" "Feroz Nakhcheer" "Roi-Do-Ab" "Dara-I-Soof-I-Payin" "Dara-I-Soof-I-Bala" "Balkh (Mazar-I-Sharif)" "Nahri Shahi" "Dehdadi" "Char Kent" "Marmul" "Balkh" "Sholgara" "Chimtal" "Dawlat Abad" "Khulm" "Char Bolak" "Shortepa" "Kaldar" "Kishindeh" "Zari" "Sharak-e Hairatan" "Sar-I-Pul (Sar-I-Pul)" "Sayyad" "Kohistanat" "Sozma Qala" "Sangcharak" "Gosfandi" "Balkhab" "Ghor (Chighcheran)" "Duleena" "Dawlatyar" "Char Sada" "Pasaband" "Shahrak" "Lal Wa Sarjangal" "Taywara" "Tulak" "Saghar" "Daykundi (Nili)" "Shahristan" "Gezab" "Ishterlai" "Khedir" "Geti" "Miramor" "Sang-I-Takht" "Kejran" "Urozgan (Tirinkot)" "Dehraoud" "Chora" "Shahidhassas" "Khas Urozgan" "Chenarto" "Zabul (Qalat)" "Tarank Wa Jaldak" "Shinkai" "Mizan" "Arghandab" "Shah Joi" "Day Chopan" "Atghar" "Naw Bahar" "Shemel Zayi" "Kakar" "Kandahar" "Arghandab" "Daman" "Panjwayee" "Zhire" "Shah Wali Kott" "Khakrez" "Arghistan" "Ghorak" "Maiwand" "Spin Boldak" "Nesh" "Miyanishin" "Shorabak" "Maruf" "Reg (Shiga)" "Dand" "Jawzjan (Sheberghan)" "Khwaja Dukoh" "Khanaqa" "Mingajik" "Qush Tepa" "Khamyab" "Aqchah" "Faizabad" "Mardyan" "Qarqin" "Darzab" "Markazi Faryab - Maymana" "Pashtun Kot" "Khwaja Sabz Poshi Wali" "Almar" "Bilchiragh" "Shirin Tagab" "Qaisar" "Gurziwan" "Dawlaitabad" "Kohistan" "Qaram Qul" "Qurghan" "Andkhoy" "Khani Charbagh" "Helmand (Lashkargah)" "Nad Ali" "Nawa-I- Barikzayi" "Nahr-I-Saraj" "Washer" "Garm Ser" "Nawzad" "Sangin Qala" "Musa Qala" "Kajaki" "Reg-I- Khan Nishin" "Baghran" "Dishu" "Marja" "Badghis (Qala-I-Now)" "Ab Kamari" "Muqur" "Qadis" "Bala Murghab" "Jawand" "Ghormach" "Herat (Herat)" "Enjil" "Nizam-I-Shahid (Guzara)" "Karrukh" "Zendajan" "Pashtun Zarghun" "Kushk (Rubat-I-Sangi)" "Gulran" "Adraskan" "Kushk-I-Kuhna" "Ghoryan" "Obe" "Kohsan" "Shindand" "Fersi" "Chishti Sharif" "Farah" "Pushtrud" "Khak-I-Safed" "Qala-I-Kah" "Shibkoh" "Bala Buluk" "Anar Dara" "Bakwa" "Lash-I-Juwayn" "Gulistan" "Pur Chaman" "Nimroz (Zaranj)" "Kang" "Asl-I-Chakhansur" "Char Burjak" "Khashrod" "Dilaram" "Pato"" ;

#delimit cr

cap drop distid 
gen distid=. 

loc n: word count `dist'
forv i = 1/`n' {
	loc a: word `i' of `id'
	loc b: word `i' of `dist' 
    qui: replace distid= `a' if districtg== "`b'"
}

/* Some districts from different provinces have the same name. To distinguish
	these same-named districts after the above operation, we separate them 
	according to the parent province. */
replace distid=2605 if districtg=="Arghandab" & provinceg=="Zabul"	
replace distid=1706 if districtg=="Baharak" & provinceg=="Badakhshan"	
replace distid=408 if districtg=="Jaghatu" & provinceg=="Wardak"
replace distid=1209 if districtg=="Jani Khel" & provinceg=="Paktika"
replace distid=1708 if districtg=="Kohistan" & provinceg=="Badakhshan"
replace distid=1115 if districtg=="Muqur" & provinceg=="Ghazni"
replace distid=114 if districtg=="Qara Bagh" & provinceg=="Kabul"
replace distid=115 if districtg=="Surubi" & provinceg=="Kabul"
replace distid=206 if districtg=="Tagab" & provinceg=="Kapisa"

save "Geographic_Data\Elevation.dta", replace
clear

*** POPULATION DATA
* Import and keep total population data for 2013-14 (i.e., Afghan year 1392)
clear
import delimited "Population_Data\cso_district_population_estimates_2004-2020.csv" 
keep if solar_year==1392
rename (province_name_eng district_name_eng district_code total_population) ///
	(provincep districtp distid pop_1314)
keep provincep districtp distid pop_1314

* Make district IDs consistent with Election data district IDs
replace distid=2404 if districtp=="Ishterlai"
replace distid=2405 if districtp=="Khedir"
replace distid=2406 if districtp=="Giti"
replace distid=2407 if districtp=="Miramor"
replace distid=2408 if districtp=="Sang -e- Takht"
replace distid=2409 if districtp=="Kejran"
replace distid=2403 if districtp=="Gizab"
replace distid=2506 if districtp=="Chinarto"

* Aggregate temporary district population into parent district population
xcollapse (sum) pop_1314 (first) provincep districtp, by(distid) ///
	norestore

* Label variables
lab var provincep "Province"
lab var districtp "District"
lab var distid "District ID"
lab var pop_1314 "Population (2013-14)"

save "Population_Data\Afghan_Population_1314.dta", replace
clear

********************************************************************************
*** DATA MERGING
********************************************************************************

/// First Election
use "Election_Data\2014_First Round Election Results.dta" , clear // Election data

merge 1:1 distid using "Population_Data\Afghan_Population_1314.dta" // Pop. data

* Drop data-checking jursdiction variables
/* Before dropping the following variables, we visually inspected the quality of
	the district-level match by entering the following command: 
		browse province* district* distid _merge */ 
drop provincep districtp _merge

merge 1:1 distid using "Violence_Data\AfgSigacts_2014_district.dta", ///
	keepusing(*_1 districtv) // Violence data

* Replace districts w/0 reported violence to 0 events for observation period	
replace sigact_5_1=0 if sigact_5_1==.
replace sigact_60_1=0 if sigact_60_1==.

* Drop data-checking jursdiction variables
/* Before dropping the following variables, we visually inspected the quality of
	the district-level match by entering the following command: 
		browse province* district* distid _merge */ 
drop districtv _merge
	
merge 1:1 distid using "Development_Data\ALCS_1314.dta" // Development data
	
* Drop data-checking jursdiction variables
/* Before dropping the following variables, we visually inspected the quality of
	the district-level match by entering the following command: 
		browse province* district* distid _merge */ 
drop provinced districtd _merge		
	
merge 1:1 distid using "Geographic_Data\Elevation.dta", ///
	keepusing(provinceg districtg elevation) // Geographic data
drop if _merge==2 // Temporary district of Pato not in election data
	
* Drop data-checking jursdiction variables
/* Before dropping the following variables, we visually inspected the quality of
	the district-level match by entering the following command: 
		browse province* district* distid _merge */ 
drop provinceg districtg _merge			

replace elect=1 if elect==.
	
save "Election_Data\AFG_FirstElection_2014", replace
	
/// Runoff Election
clear
use "Election_Data\2014_Runoff Election Results.dta" // Election data

merge 1:1 distid using "Population_Data\Afghan_Population_1314.dta" // Pop. data

* Drop data-checking jursdiction variables
/* Before dropping the following variables, we visually inspected the quality of
	the district-level match by entering the following command: 
		browse province* district* distid _merge */ 
drop provincep districtp _merge

merge 1:1 distid using "Violence_Data\AfgSigacts_2014_district.dta", ///
	keepusing(*_2 districtv) // Violence data

* Replace districts w/0 reported violence to 0 events for observation period	
replace sigact_5_2=0 if sigact_5_2==.
replace sigact_60_2=0 if sigact_60_2==.

* Drop data-checking jursdiction variables
/* Before dropping the following variables, we visually inspected the quality of
	the district-level match by entering the following command: 
		browse province* district* distid _merge */ 
drop districtv _merge
	
merge 1:1 distid using "Development_Data\ALCS_1314.dta" // Development data
	
* Drop data-checking jursdiction variables
/* Before dropping the following variables, we visually inspected the quality of
	the district-level match by entering the following command: 
		browse province* district* distid _merge */ 
drop provinced districtd _merge		
	
merge 1:1 distid using "Geographic_Data\Elevation.dta", ///
	keepusing(provinceg districtg elevation) // Geographic data
drop if _merge==2 // Temporary district of Pato not in election data
	
* Drop data-checking jursdiction variables
/* Before dropping the following variables, we visually inspected the quality of
	the district-level match by entering the following command: 
		browse province* district* distid _merge */ 
drop provinceg districtg _merge		

replace elect=2 if elect==.	
	
save "Election_Data\AFG_RunnoffElection_2014", replace
	
/// Append first and runoff elections
append using "Election_Data\AFG_FirstElection_2014"

* Combine violence variables
replace sigact_5_1=sigact_5_2 if sigact_5_1==.
replace sigact_60_1=sigact_60_2 if sigact_60_1==.		
drop sigact_5_2-sigact_60_2
rename (sigact_5_1 sigact_60_1)	(sigact_5 sigact_60)
lab var sigact_5 "Violent Events (Sigact-5 days)"
lab var sigact_60 "Violent Events (Sigact-60 days)"

sort distid elect

* Clean data
replace votes=0 if votes==.

* Retain district observations only if election fraud outcome is observed
drop if fraud==. // N=16 observations deleted; N=800 retained

* Drop unnecessary measures for final analytic data file
drop psn psno psnx psx

/// Create final measures

* Violence rate
gen sigact_5r=sigact_5/(pop_1314/1000)
lab var sigact_5r "Violent Incident Rate/1,000 Population (5-Day Window)"
gen sigact_60r=sigact_60/(pop_1314/1000)
lab var sigact_60r "Violent Incident Rate/1,000 Population (60-Day Window)"

* Expenditures
gen pcexpend=(tot_expend/1000)/pop_1314
lab var pcexpend "Per Capita Monthly Expenditure"
order pcexpend, a(tot_expend)

* Electrification
gen electricp=electric*100
lab var electricp "% Electrification"
order electricp, a(electric)

* Elevation
/* Elevation data are missing for 9 districts. We recovered these data by 
	manually searching latitude and longitude coordinates in Google Earth 
	(earth.google.com) and entering the results here. */
replace elevation=1339 if distid==706
replace elevation=2763 if distid==808
replace elevation=2364 if distid==1312
replace elevation=2520 if distid==1313
replace elevation=1779 if distid==2506
replace elevation=985 if distid==2717
replace elevation=1036 if distid==2718
replace elevation=767 if distid==3014
replace elevation=812 if distid==3406

gen elevationk=elevation/1000 // elevation in kilometers
lab var elevationk "Average District Elevation (km)"
order elevationk, a(elevation)

* Distance to Kabul
* Create reference coordinates for Kabul
gen klat=34.52721
gen klon=69.16262
geodist lat lon klat klon, gen(dist) sphere
lab var dist "Distance to Kabul (km)"
drop klat klon
order dist, a(elevationk)

* Regional command (Clustering variable)
encode province, gen(provid)
recode provid (1 3 4 8 13 19 29/31=1 "RC-North") ///
	(2 7 10 12=2 "RC-West") (14=3 "RC-Central") ///
	(5 9 16/18 20/22 24/28 33=4 "RC-East") ///
	(11 23=5 "RC-Southwest") (6 15 32 34=6 "RC-South"), gen(regcom) test
lab var regcom "Regional Commands"
drop provid

* Format
format pcx sigact_5r sigact_60r pcexpend elevationk %6.2fc
format fraud votes pop_1314 %11.0fc
format electric %6.2fc
format electricp dist %6.1fc

label define fraud 0 "No Fraud" 1 "Fraud"
label define elect 1 "1st Election" 2 "2nd Election"
label values fraud fraud
label values elect elect

compress

save "Afghanistan_Election_Violence_2014.dta", replace	

// End data management