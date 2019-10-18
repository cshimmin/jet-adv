ll = np.load('preds_adv_LL.npz')
hl = np.load('preds_adv_HL.npz')

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
category_styles = [Line2D([0], [0], color='blue', label='background'),
                Line2D([0], [0], color='darkorange', label='signal'),
               ]
adv_styles = [Patch(facecolor='gray', alpha=0.2, label='unperturbed'),
              Line2D([0],[0], color='gray', label='HL perturbed', ls='--'),
              Line2D([0],[0], color='gray', label='LL pertrubed'),
               ]

density=True

figsize=(8,4)

plt.figure(figsize=figsize)
b = np.linspace(0,1,21)
plt.hist(hl['preds_val'][y_val==0,0], color='blue', bins=b, alpha=0.2, density=density);
plt.hist(hl['preds_val'][y_val==1,0], color='darkorange', bins=b, alpha=0.2, density=density)
plt.hist(hl['preds_adv'][y_val==0,0], color='blue', bins=b, histtype='step', ls='--', density=density)
plt.hist(hl['preds_adv'][y_val==1,0], color='darkorange', bins=b, histtype='step', ls='--', density=density);

plt.yticks([0,])

l1 = plt.legend(handles=category_styles, loc='upper center', ncol=2)
plt.gca().add_artist(l1)
plt.legend(handles=adv_styles, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.)

plt.xlabel("HL classifier response")
plt.ylabel("A.U.")

plt.savefig('adv_compare_HL_cls.pdf', bbox_inches='tight')

plt.figure(figsize=figsize)
b = np.linspace(0,1,21)
plt.hist(ll['preds_val'][y_val==0,0], bins=b, color='blue', alpha=0.2, density=density);
plt.hist(ll['preds_val'][y_val==1,0], bins=b, color='darkorange', alpha=0.2, density=density)
plt.hist(ll['preds_adv'][y_val==0,0], bins=b, color='blue', histtype='step', density=density)
plt.hist(ll['preds_adv'][y_val==1,0], bins=b, color='darkorange', histtype='step', density=density);

plt.yticks([0,])

l1 = plt.legend(handles=category_styles, loc='upper center', ncol=2)
plt.gca().add_artist(l1)
plt.legend(handles=adv_styles, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.)

plt.xlabel("LL classifier response")
plt.ylabel("A.U.")

plt.savefig('adv_compare_LL_cls.pdf', bbox_inches='tight')


plt.figure(figsize=figsize)
b = np.linspace(350,525,21)
plt.hist(hl['jpt_val'][y_val==0]*1e3, bins=b, color='blue', alpha=0.2, density=density);
plt.hist(hl['jpt_val'][y_val==1]*1e3, bins=b, color='darkorange', alpha=0.2, density=density)
plt.hist(hl['jpt_adv'][y_val==0]*1e3, bins=b, color='blue', histtype='step', ls='--', density=density);
plt.hist(hl['jpt_adv'][y_val==1]*1e3, bins=b, color='darkorange', histtype='step', ls='--', density=density);
plt.hist(ll['jpt_adv'][y_val==0]*1e3, bins=b, color='blue', histtype='step', density=density);
plt.hist(ll['jpt_adv'][y_val==1]*1e3, bins=b, color='darkorange', histtype='step', density=density);

plt.yticks([0,])

l1 = plt.legend(handles=category_styles, loc='upper right', ncol=2)
plt.gca().add_artist(l1)
plt.legend(handles=adv_styles, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.)

plt.xlabel(r"Jet $p_\mathrm{T}$ [GeV]")
plt.ylabel("A.U.")

plt.savefig('adv_compare_jpt_cls.pdf', bbox_inches='tight')


plt.figure(figsize=figsize)
b = np.linspace(60,220,21)
plt.hist(hl['jmass_val'][y_val==0]*1e3, bins=b, color='blue', alpha=0.2, density=density);
plt.hist(hl['jmass_val'][y_val==1]*1e3, bins=b, color='darkorange', alpha=0.2, density=density)
plt.hist(hl['jmass_adv'][y_val==0]*1e3, bins=b, color='blue', histtype='step', ls='--', density=density);
plt.hist(hl['jmass_adv'][y_val==1]*1e3, bins=b, color='darkorange', histtype='step', ls='--', density=density);
plt.hist(ll['jmass_adv'][y_val==0]*1e3, bins=b, color='blue', histtype='step', density=density);
plt.hist(ll['jmass_adv'][y_val==1]*1e3, bins=b, color='darkorange', histtype='step', density=density);

plt.yticks([0,])

l1 = plt.legend(handles=category_styles, loc='upper right', ncol=2)
plt.gca().add_artist(l1)
plt.legend(handles=adv_styles, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.);

plt.xlabel(r"Jet mass [GeV]")
plt.ylabel("A.U.");

plt.savefig('adv_compare_jmass_cls.pdf', bbox_inches='tight')


plt.figure(figsize=figsize)
b = np.linspace(0,10,21)
d2scale=1.
plt.hist(hl['jd2_val'][y_val==0]*d2scale, bins=b, color='blue', alpha=0.2, density=density);
plt.hist(hl['jd2_val'][y_val==1]*d2scale, bins=b, color='darkorange', alpha=0.2, density=density)
plt.hist(hl['jd2_adv'][y_val==0]*d2scale, bins=b, color='blue', histtype='step', ls='--', density=density);
plt.hist(hl['jd2_adv'][y_val==1]*d2scale, bins=b, color='darkorange', histtype='step', ls='--', density=density);
plt.hist(ll['jd2_adv'][y_val==0]*d2scale, bins=b, color='blue', histtype='step', density=density);
plt.hist(ll['jd2_adv'][y_val==1]*d2scale, bins=b, color='darkorange', histtype='step', density=density);

plt.yticks([0,])

l1 = plt.legend(handles=category_styles, loc='upper right', ncol=2)
plt.gca().add_artist(l1)
plt.legend(handles=adv_styles, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.);

plt.xlabel(r"Jet $D_2^{(\beta=2)}$")
plt.ylabel("A.U.");

plt.savefig('adv_compare_jd2_cls.pdf', bbox_inches='tight')




####
# plots for relative signifiance
####
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='gray'),
                Line2D([0], [0], color='gray', ls='--'),
                Line2D([0], [0], color='gray', ls=':'),
               ]


plt.figure(figsize=(8,4))

l1, = plt.plot(thresholds, pfn_sig_exp, color='darkorange', label=r'Low-level')
plt.plot(thresholds, pfn_sig_obs, color='darkorange', ls='--')
plt.plot(thresholds, pfn_sig_rnd, color='darkorange', ls=':')

plt.axvline(thresholds[np.argmax(pfn_sig_exp)], ls='-', color='slategray', zorder=-1, lw=1)

l2, = plt.plot(thresholds, ut_sig_exp, color='red', label=r'Low-level (undertrained)')
plt.plot(thresholds, ut_sig_obs, color='red', ls='--')
plt.plot(thresholds, ut_sig_rnd, color='red', ls=':')


l4, = plt.plot(thresholds, hl_sig_exp, color='blue', label='High-level')
plt.plot(thresholds, hl_sig_obs, color='blue', ls='--')
plt.plot(thresholds, hl_sig_rnd, color='blue', ls=':')

l3, = plt.plot(thresholds, ut2_sig_exp, color='mediumseagreen', label=r'Low-level (undertrained)')
plt.plot(thresholds, ut2_sig_obs, color='mediumseagreen', ls='--')
plt.plot(thresholds, ut2_sig_rnd, color='mediumseagreen', ls=':')

plt.ylabel("Relative Discovery Significance")
plt.xlabel("Classifier Threshold")
first_legend = plt.legend(handles=[l1,l2,l3,l4], loc='upper left');
plt.gca().add_artist(first_legend)
plt.legend(custom_lines, ['Expected','Observed', 'Random'], loc='lower center', ncol=3);
plt.savefig("signif_undertraining.pdf", bbox_inches='tight')
