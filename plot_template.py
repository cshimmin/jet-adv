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

