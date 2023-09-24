#
# stations used:
STATIONS = 185,157,259,319,122,205,143,204,244,243,305,228,211,308,310,276,167,240,236,263,238,275,264,214,189,316,262,209,207,125,176,268,306,191,133,202,300,129,195,149,160,201,290,255,224,296,123,128,269,302,147,179,169,261,295,131,309,150,297,294,246,223,152,161,158,190,168,286,183,272,144,273,148,318,299,165,210,226,304,287,280,292,192,170,307,229,127,285,284,233,225,216,173,256,271,267,314,288,247,250

# t_bins and f_bins of spectrogram
t = [0.0, 0.1040000063476566, 0.2080000126953132, 0.3120000190429699, 0.4160000253906265, 0.5200000317382831, 0.6240000380859398, 0.7280000444335963, 0.832000050781253, 0.9360000571289098, 1.0400000634765663, 1.144000069824223, 1.2480000761718797, 1.3520000825195362, 1.456000088867193, 1.5600000952148496, 1.6640001015625063, 1.7680001079101626, 1.8720001142578193, 1.976000120605476, 2.0800001269531325, 2.1840001333007892, 2.288000139648446, 2.3920001459961027, 2.4960001523437594, 2.600000158691416, 2.704000165039073, 2.8080001713867295, 2.9120001777343862, 3.016000184082042, 3.120000190429699, 3.2240001967773555, 3.328000203125012, 3.432000209472669, 3.5360002158203256, 3.6400002221679824, 3.744000228515639, 3.848000234863295, 3.9520002412109516, 4.056000247558608]
f = [0.9765624403953552, 1.9531248807907104, 2.9296873211860657, 3.906249761581421, 4.882812201976776, 5.859374642372131, 6.835937082767487, 7.812499523162842, 8.789061963558197, 9.765624403953552, 10.742186844348907, 11.718749284744263, 12.695311725139618, 13.671874165534973, 14.648436605930328, 15.624999046325684, 16.60156148672104, 17.578123927116394, 18.55468636751175, 19.531248807907104, 20.50781124830246, 21.484373688697815, 22.46093612909317, 23.437498569488525, 24.41406100988388, 25.390623450279236, 26.36718589067459, 27.343748331069946, 28.3203107714653, 29.296873211860657, 30.273435652256012, 31.249998092651367, 32.22656053304672, 33.20312297344208, 34.17968541383743, 35.15624785423279, 36.13281029462814, 37.1093727350235, 38.085935175418854, 39.06249761581421, 40.039060056209564, 41.01562249660492, 41.992184937000275, 42.96874737739563, 43.945309817790985, 44.92187225818634, 45.898434698581696, 46.87499713897705, 47.851559579372406, 48.82812201976776, 49.80468446016312, 50.78124690055847, 51.75780934095383, 52.73437178134918, 53.71093422174454, 54.68749666213989, 55.66405910253525, 56.6406215429306, 57.61718398332596, 58.59374642372131, 59.57030886411667, 60.546871304512024, 61.52343374490738, 62.499996185302734, 63.47655862569809, 64.45312106609344, 65.4296835064888, 66.40624594688416, 67.38280838727951, 68.35937082767487, 69.33593326807022, 70.31249570846558, 71.28905814886093, 72.26562058925629, 73.24218302965164, 74.218745470047, 75.19530791044235, 76.17187035083771, 77.14843279123306, 78.12499523162842, 79.10155767202377, 80.07812011241913, 81.05468255281448, 82.03124499320984, 83.0078074336052, 83.98436987400055, 84.9609323143959, 85.93749475479126, 86.91405719518661, 87.89061963558197, 88.86718207597733, 89.84374451637268, 90.82030695676804, 91.79686939716339, 92.77343183755875, 93.7499942779541, 94.72655671834946, 95.70311915874481, 96.67968159914017, 97.65624403953552, 98.63280647993088, 99.60936892032623, 100.58593136072159, 101.56249380111694, 102.5390562415123, 103.51561868190765, 104.49218112230301, 105.46874356269836, 106.44530600309372, 107.42186844348907, 108.39843088388443, 109.37499332427979, 110.35155576467514, 111.3281182050705, 112.30468064546585, 113.2812430858612, 114.25780552625656, 115.23436796665192, 116.21093040704727, 117.18749284744263, 118.16405528783798, 119.14061772823334, 120.11718016862869, 121.09374260902405, 122.0703050494194, 123.04686748981476, 124.02342993021011, 124.99999237060547, 125.97655481100082, 126.95311725139618, 127.92967969179153, 128.9062421321869, 129.88280457258224, 130.8593670129776, 131.83592945337296, 132.8124918937683, 133.78905433416367, 134.76561677455902, 135.74217921495438, 136.71874165534973, 137.6953040957451, 138.67186653614044, 139.6484289765358, 140.62499141693115, 141.6015538573265, 142.57811629772186, 143.55467873811722, 144.53124117851257, 145.50780361890793, 146.48436605930328, 147.46092849969864, 148.437490940094, 149.41405338048935, 150.3906158208847, 151.36717826128006, 152.34374070167542, 153.32030314207077, 154.29686558246613, 155.27342802286148, 156.24999046325684, 157.2265529036522, 158.20311534404755, 159.1796777844429, 160.15624022483826, 161.1328026652336, 162.10936510562897, 163.08592754602432, 164.06248998641968, 165.03905242681503, 166.0156148672104, 166.99217730760574, 167.9687397480011, 168.94530218839645, 169.9218646287918, 170.89842706918716, 171.87498950958252, 172.85155194997787, 173.82811439037323, 174.80467683076859, 175.78123927116394, 176.7578017115593, 177.73436415195465, 178.71092659235, 179.68748903274536, 180.66405147314072, 181.64061391353607, 182.61717635393143, 183.59373879432678, 184.57030123472214, 185.5468636751175, 186.52342611551285, 187.4999885559082, 188.47655099630356, 189.4531134366989, 190.42967587709427, 191.40623831748962, 192.38280075788498, 193.35936319828033, 194.3359256386757, 195.31248807907104, 196.2890505194664, 197.26561295986176, 198.2421754002571, 199.21873784065247, 200.19530028104782, 201.17186272144318, 202.14842516183853, 203.1249876022339, 204.10155004262924, 205.0781124830246, 206.05467492341995, 207.0312373638153, 208.00779980421066, 208.98436224460602, 209.96092468500137, 210.93748712539673, 211.91404956579208, 212.89061200618744, 213.8671744465828, 214.84373688697815, 215.8202993273735, 216.79686176776886, 217.77342420816422, 218.74998664855957, 219.72654908895493, 220.70311152935028, 221.67967396974564, 222.656236410141, 223.63279885053635, 224.6093612909317, 225.58592373132706, 226.5624861717224, 227.53904861211777, 228.51561105251312, 229.49217349290848, 230.46873593330383, 231.4452983736992, 232.42186081409454, 233.3984232544899, 234.37498569488525, 235.3515481352806, 236.32811057567596, 237.30467301607132, 238.28123545646667, 239.25779789686203, 240.23436033725739, 241.21092277765274, 242.1874852180481, 243.16404765844345, 244.1406100988388, 245.11717253923416, 246.09373497962952, 247.07029742002487, 248.04685986042023, 249.02342230081558, 249.99998474121094]
