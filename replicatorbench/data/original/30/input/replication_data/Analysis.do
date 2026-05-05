use "...\Full_long.dta"
  generate nvst = . 
    replace nvst = playerinvestment if round > 2
  generate pstn = .
    replace pstn = 1 if playerrole == "A"
	replace pstn = 2 if playerrole == "B"
	replace pstn = 3 if playerrole == "C"
	replace pstn = 4 if playerrole == "D"
	replace pstn = 5 if playerrole == "E"
  generate rnd = .
    replace rnd = round - 2 if round > 2
  generate dwnld = .	
    replace dwnld = playerdownloaded_files if round > 2
  generate pyff = .
    replace pyff = playercollected_tokens + (10 - playerinvestment) if round > 2
  generate grp = Session*100+groupid_in_subsession
  generate idrnd = id*100+rnd
  generate grprnd = grp*100+rnd
  generate dwnldpyff = .	
    replace dwnldpyff = playercollected_tokens if round > 2
  generate bndwdth = .	
    replace bndwdth = groupbandwidth if round > 2
save "...\Full_long_v1.dta"
use "...\Full_long_v1.dta"
  drop if round < 3
  collapse (sum) nvst dwnldpyff, by(grprnd)
    rename nvst grpnvst
	rename dwnldpyff grpdwnldpyff
save "...\Sums.dta"
use "...\Full_long_v1.dta"
  drop if round < 3
  sort idrnd
  merge m:1 grprnd using "...\Sums.dta"
  generate shrnvst = .
     replace shrnvst = nvst/grpnvst if grpnvst > 0
	 replace shrnvst = 0.2 if grpnvst == 0 
  generate shrdwnld = .
     replace shrdwnld = dwnldpyff/grpdwnldpyff if grpdwnldpyff > 0
	 replace shrdwnld = 0.2 if grpdwnldpyff == 0 
  xtset id rnd
  generate shr = L1.shrdwnld
  generate pstnshr = pstn * shr
  generate pstnshrnvst = pstn * shrnvst
save "...\Full_long_v2.dta"
use "...\Full_long_v2.dta"
  summarize nvst if pstn == 1
  summarize nvst if pstn == 2
  summarize nvst if pstn == 3
  summarize nvst if pstn == 4
  summarize nvst if pstn == 5
  summarize dwnld if pstn == 1
  summarize dwnld if pstn == 2
  summarize dwnld if pstn == 3
  summarize dwnld if pstn == 4
  summarize dwnld if pstn == 5
  summarize pyff if pstn == 1
  summarize pyff if pstn == 2
  summarize pyff if pstn == 3
  summarize pyff if pstn == 4
  summarize pyff if pstn == 5
  mixed nvst pstn shr pstnshr rnd || grp:
  mixed dwnldpyff bndwdth pstn shrnvst pstnshrnvst rnd || grp:
  mixed nvst b(1).pstn##c.shr b(10).rnd || grp:
  mixed nvst b(1).pstn##c.shr b(10).rnd || grp:, reml
  mixed dwnldpyff bndwdth b(1).pstn##c.shrnvst b(10).rnd || grp:
  mixed dwnldpyff bndwdth b(1).pstn##c.shrnvst b(10).rnd || grp:, reml
  