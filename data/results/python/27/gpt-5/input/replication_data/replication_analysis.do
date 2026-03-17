clear
insheet using ".\round1_raw.csv"
save round1raw

keep if ipo_task1groupid_in_subsession>1

rename ipo_task1groupmarket_price price1
rename ipo_task2groupmarket_price price2
rename ipo_task3groupmarket_price price3
rename ipo_task4groupmarket_price price4
rename ipo_task5groupmarket_price price5
rename ipo_task6groupmarket_price price6
rename ipo_task7groupmarket_price price7
rename ipo_task8groupmarket_price price8
rename ipo_task9groupmarket_price price9
rename ipo_task10groupmarket_price price10
rename ipo_task11groupmarket_price price11
rename ipo_task12groupmarket_price price12
rename ipo_task13groupmarket_price price13
rename ipo_task14groupmarket_price price14
rename ipo_task15groupmarket_price price15
rename ipo_task16groupmarket_price price16
rename ipo_task17groupmarket_price price17
rename ipo_task18groupmarket_price price18
rename ipo_task19groupmarket_price price19
rename ipo_task20groupmarket_price price20

rename ipo_task20playertotal_missing_re rounds_missed


reshape long price, i(ipo_task1groupid_in_subsession ipo_task1playerid_in_group sessioncode rounds_missed) j(round)

egen group = group(ipo_task1groupid_in_subsession sessioncode)

collapse (mean) price=price (max) rounds_missed_max=rounds_missed (min) rounds_missed_min=rounds_missed, by(group round)

gen dropout=0
replace dropout=1 if rounds_missed_max>=6

gen bankrupt=0
replace bankrupt=1 if rounds_missed_min==-99

signrank price=1.94
signrank price=1.94 if !dropout
signrank price=1.94 if !dropout&!bankrupt

collapse (mean) price=price (max) dropout bankrupt, by(group)

signrank price=1.94
signrank price=1.94 if !dropout
signrank price=1.94 if !dropout&!bankrupt

outsheet using round1_analysis.csv, comma

clear
insheet using ".\round2_raw.csv"
append using round1raw

keep if ipo_task1groupid_in_subsession>1

rename ipo_task1groupmarket_price price1
rename ipo_task2groupmarket_price price2
rename ipo_task3groupmarket_price price3
rename ipo_task4groupmarket_price price4
rename ipo_task5groupmarket_price price5
rename ipo_task6groupmarket_price price6
rename ipo_task7groupmarket_price price7
rename ipo_task8groupmarket_price price8
rename ipo_task9groupmarket_price price9
rename ipo_task10groupmarket_price price10
rename ipo_task11groupmarket_price price11
rename ipo_task12groupmarket_price price12
rename ipo_task13groupmarket_price price13
rename ipo_task14groupmarket_price price14
rename ipo_task15groupmarket_price price15
rename ipo_task16groupmarket_price price16
rename ipo_task17groupmarket_price price17
rename ipo_task18groupmarket_price price18
rename ipo_task19groupmarket_price price19
rename ipo_task20groupmarket_price price20

rename ipo_task20playertotal_missing_re rounds_missed


reshape long price, i(ipo_task1groupid_in_subsession ipo_task1playerid_in_group sessioncode rounds_missed) j(round)

egen group = group(ipo_task1groupid_in_subsession sessioncode)

collapse (mean) price=price (max) rounds_missed_max=rounds_missed (min) rounds_missed_min=rounds_missed, by(group round)

gen dropout=0
replace dropout=1 if rounds_missed_max>=6

gen bankrupt=0
replace bankrupt=1 if rounds_missed_min==-99

signrank price=1.94
signrank price=1.94 if !dropout
signrank price=1.94 if !dropout&!bankrupt

collapse (mean) price=price (max) rounds_missed_max dropout bankrupt, by(group)

signrank price=1.94
signrank price=1.94 if !dropout
signrank price=1.94 if !dropout&!bankrupt

tab rounds_missed_max

outsheet using fullsample_analysis.csv, comma
