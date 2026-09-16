#!/usr/bin/env bash
cd "$(dirname "$0")"
exec ../../.venv/bin/python3 ens_select_cv.py --jobs 90 --out ens_sets_cv.json --sets \
 "q_endovis18_dlv3" \
 "q_endovis18_dlv3,q_cholec_dlv3,r_toolpaste,r_xl" \
 "q_endovis18_dlv3,q_cholec_dlv3,r_toolpaste,r_xl,e_convnext_xl_384,k_dicedet_rules" \
 "q_endovis18_dlv3,q_cholec_dlv3,r_toolpaste,r_xl,e_convnext_xl_384,k_dicedet_rules,q_both_dlv3,l_dicedet" \
 "q_endovis18_dlv3,q_cholec_dlv3,r_toolpaste,r_xl,e_convnext_xl_384,k_dicedet_rules,q_both_dlv3,l_dicedet,r_detbd,q_evshallow_dlv3" \
 "q_endovis18_dlv3,q_cholec_dlv3,r_toolpaste,r_xl,e_convnext_xl_384,k_dicedet_rules,q_both_dlv3,l_dicedet,r_detbd,q_evshallow_dlv3,k_rules_boundary,a_nohflip,l_rules,s_kdr_seed43"
