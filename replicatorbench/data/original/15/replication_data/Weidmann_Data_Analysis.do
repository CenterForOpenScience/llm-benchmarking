********************************************************************************
*		WEIDMANN & CALLEN (2013) REPLICATION 
* 		DARPA SCORE PROJECT: 
*		"Replication of a Research Claim from Weidmann & Callen (2013)"
*		Eric L. Sevigny & Jared Greathouse
* 		Contact: esevigny@gsu.edu
********************************************************************************

********************************************************************************
** DATA ANALYSIS
********************************************************************************

/// Users should insert project directory between quotes before running code:
cap cd "INSERT PROJECT DIRECTORY HERE" 

/// Stata MP version
version 16.1

use "Afghanistan_Election_Violence_2014.dta", clear

/// Install Commands
net install estout.pkg, from(http://fmwww.bc.edu/RePEc/bocode/e)
net install spost13_ado, from(https://jslsoc.sitehost.iu.edu/stata)
net install gr0070, from(http://www.stata-journal.com/software/sj17-3)
net install st0582_1, from(http://www.stata-journal.com/software/sj20-2)

/// Analyses	
* Relabel variables to match original study
lab var fraud "Fraud, last digit test"
lab var sigact_5r "Violence (election)"
lab var sigact_60r "Violence (2 months, pre-election)"
lab var pcx "Percentage of centers closed"
lab var electric "Electrification"
lab var pcexpend "Per-capita expenditure (1000 AFs)"
lab var dist "Distance from Kabul (km)"
lab var elevation "Elevation (m)"

// Reproduce Original Table 1 of Summary Statistics 
eststo clear
qui: estpost sum fraud pcx sigact_5r sigact_60r pcexpend electric dist elevation

esttab using 1Table1.rtf, label nonum noobs replace compress nomtit nogaps ///
	ti("Table 1 Summary Statistics for the Variables Included in the Regression Analysis") ///
	cells("mean(label(Mean) fmt(2)) sd(label(Std. Dev.) fmt(2)) min(label(Min)) max(label(Max)) count(label(N) fmt(a2))") ///
	refcat(fraud "\i Fraud \i0" sigact_5r "\i Violence \i0" pcexpend "\i Development \i0" dist "\i Geography \i0", nolabel)
	
// Reproduce Table 2, Model 1 Regression of Election Fraud on Violence, 5-Day Window
eststo clear
eststo m2_1: logit fraud c.sigact_5r##c.sigact_5r pcx electric pcexpend dist ///
	elevation,	vce(cluster regcom)
estadd fitstat // Obtain McFadden's R2

* Graph Results
margins, at(sigact_5r=(0(.01).40))
marginsplot, scheme(plotplain) saving(m2_1, replace) title("") ///
	xtitle("(1) 5-Day Election Window (One-Way Clustering)") ytitle("")

// Reproduce Table 2, Model 1: Supplemental Analysis Using Two-Way Clustering
eststo m2_1s: vcemway logit fraud c.sigact_5r##c.sigact_5r pcx electric pcexpend dist ///
	elevation,	cluster(regcom elect)
matrix list r(table) // Report exact p-values 
estadd fitstat // Obtain McFadden's R2

margins, at(sigact_5r=(0(.01).40))
marginsplot, scheme(plotplain) saving(m2_1s, replace) title("") ///
	xtitle("(3) 5-Day Election Window (Multiway Clustering)") ytitle("")	
	
// Reproduce Table 2, Model 2 Regression of Election Fraud on Violence, 60-Day Window
eststo m2_2: logit fraud c.sigact_60r##c.sigact_60r pcx electric pcexpend dist ///
	elevation,	vce(cluster regcom)
matrix list r(table) // Report exact p-values 
estadd fitstat // Obtain McFadden's R2

* Graph Results
margins, at(sigact_60r=(0(.075)3))
marginsplot, scheme(plotplain) saving(m2_2, replace) title("") ///
	xtitle("(2) 60-Day Election Window (One-Way Clustering)") ytitle("")

// Reproduce Table 2, Model 2: Supplemental Analysis Using Two-Way Clustering
eststo m2_2s: vcemway logit fraud c.sigact_60r##c.sigact_60r pcx electric pcexpend dist ///
	elevation,	cluster(regcom elect)	
matrix list r(table) // Report exact p-values 
estadd fitstat // Obtain McFadden's R2

margins, at(sigact_60r=(0(.075)3))
marginsplot, scheme(plotplain) saving(m2_2s, replace) title("") ///
	xtitle("(4) 60-Day Election Window (Multiway Clustering)") ytitle("")
	
// Create Combined Table of Results: Report p-values
esttab m2_1 m2_2 m2_1s m2_2s using Table2p.rtf, replace compress label one interact("*") nomti ///
	s(r2_mf clustvar N, l("McFadden's \i R\i0\super 2 \super0" "Clustering" "N")) ///
	nogaps b(a3) p(a3) nostar ///
	order(sigact_5r c.sigact_5r#c.sigact_5r sigact_60r c.sigact_60r#c.sigact_60r) ///
	ti("Table 2. Logit Regressions of Election Fraud on Violence") 
	
// Create Combined Table of Results: Report SEs
esttab m2_1 m2_2 m2_1s m2_2s using Table2se.rtf, replace compress label one interact("*") nomti ///
	s(r2_mf clustvar N, l("McFadden's \i R\i0\super 2 \super0" "Clustering" "N")) ///
	nogaps b(a3) se(a3) nostar bracket ///
	order(sigact_5r c.sigact_5r#c.sigact_5r sigact_60r c.sigact_60r#c.sigact_60r) ///
	ti("Table 2. Logit Regressions of Election Fraud on Violence") 
	
// Creat Combined Graph
gr combine m2_1.gph m2_2.gph m2_1s.gph m2_2s.gph, ycommon scheme(plotplain) ///
	saving(Fraud, replace)

clear