#!/usr/bin/env python3
"""
This module containes general-purpose chemical data (aka tables).

Sources:

    1_Energy_Contrast. www.ccdc.cam.ac.uk/Lists/ResourceFileList/Elemental_Radii.xlsx, (access
    date: 13 Oct 2015)

    2. C. W. Yong, 'DL_FIELD - A force field and model development tool for
    DL_POLY', R. Blake, Ed., CSE Frontier, STFC Computational Science and
    Engineering, Daresbury Laboratory, UK, p38-40 (2010)

    3. https://www.ccdc.cam.ac.uk/support-and-resources/ccdcresources/
    Elemental_Radii.xlsx (access date: 26 Jul 2017)

Note:

Added atomic mass, atomic van der Waals radius and covalent radius, that equals
1_Energy_Contrast, for a dummy atom `X`. This is for the shape descriptors calculations to use
with functions for molecular shape descriptors that require mass.
"""

# Atomic mass dictionary (in upper case!)
atomic_mass = {
    'AL': 26.982,
    'SB': 121.76,
    'AR': 39.948,
    'AS': 74.922,
    'BA': 137.327,
    'BE': 9.012,
    'BI': 208.98,
    'B': 10.811,
    'BR': 79.904,
    'CD': 112.411,
    'CS': 132.905,
    'CA': 40.078,
    'C': 12.011,
    'CE': 140.116,
    'CL': 35.453,
    'CR': 51.996,
    'CO': 58.933,
    'CU': 63.546,
    'DY': 162.5,
    'ER': 167.26,
    'EU': 151.964,
    'F': 18.998,
    'GD': 157.25,
    'GA': 69.723,
    'GE': 72.61,
    'AU': 196.967,
    'HF': 178.49,
    'HE': 4.003,
    'HO': 164.93,
    'H': 1.008,
    'IN': 114.818,
    'I': 126.904,
    'IR': 192.217,
    'FE': 55.845,
    'KR': 83.8,
    'LA': 138.906,
    'PB': 207.2,
    'LI': 6.941,
    'LU': 174.967,
    'MG': 24.305,
    'MN': 54.938,
    'HG': 200.59,
    'MO': 95.94,
    'ND': 144.24,
    'NE': 20.18,
    'NI': 58.693,
    'NB': 92.906,
    'N': 14.007,
    'OS': 190.23,
    'O': 15.999,
    'PD': 106.42,
    'P': 30.974,
    'PT': 195.078,
    'K': 39.098,
    'PR': 140.908,
    'PA': 231.036,
    'RE': 186.207,
    'RH': 102.906,
    'RB': 85.468,
    'RU': 101.07,
    'SM': 150.36,
    'SC': 44.956,
    'SE': 78.96,
    'SI': 28.086,
    'AG': 107.868,
    'NA': 22.991,
    'SR': 87.62,
    'S': 32.066,
    'TA': 180.948,
    'TE': 127.6,
    'TB': 158.925,
    'TL': 204.383,
    'TH': 232.038,
    'TM': 168.934,
    'SN': 118.71,
    'TI': 47.867,
    'W': 183.84,
    'U': 238.029,
    'V': 50.942,
    'XE': 131.29,
    'YB': 173.04,
    'Y': 88.906,
    'ZN': 65.39,
    'ZR': 91.224,
    'X': 1,
}

# Atomic vdW radii dictionary (in upper case!)
atomic_vdw_radius = {
    'AL': 2,
    'SB': 2,
    'AR': 1.88,
    'AS': 1.85,
    'BA': 2,
    'BE': 2,
    'BI': 2,
    'B': 2,
    'BR': 1.85,
    'CD': 1.58,
    'CS': 2,
    'CA': 2,
    'C': 1.7,
    'CE': 2,
    'CL': 1.75,
    'CR': 2,
    'CO': 2,
    'CU': 1.4,
    'DY': 2,
    'ER': 2,
    'EU': 2,
    'F': 1.47,
    'GD': 2,
    'GA': 1.87,
    'GE': 2,
    'AU': 1.66,
    'HF': 2,
    'HE': 1.4,
    'HO': 2,
    'H': 1.09,
    'IN': 1.93,
    'I': 1.98,
    'IR': 2,
    'FE': 2,
    'KR': 2.02,
    'LA': 2,
    'PB': 2.02,
    'LI': 1.82,
    'LU': 2,
    'MG': 1.73,
    'MN': 2,
    'HG': 1.55,
    'MO': 2,
    'ND': 2,
    'NE': 1.54,
    'NI': 1.63,
    'NB': 2,
    'N': 1.55,
    'OS': 2,
    'O': 1.52,
    'PD': 1.63,
    'P': 1.8,
    'PT': 1.72,
    'K': 2.75,
    'PR': 2,
    'PA': 2,
    'RE': 2,
    'RH': 2,
    'RB': 2,
    'RU': 2,
    'SM': 2,
    'SC': 2,
    'SE': 1.9,
    'SI': 2.1,
    'AG': 1.72,
    'NA': 2.27,
    'SR': 2,
    'S': 1.8,
    'TA': 2,
    'TE': 2.06,
    'TB': 2,
    'TL': 1.96,
    'TH': 2,
    'TM': 2,
    'SN': 2.17,
    'TI': 2,
    'W': 2,
    'U': 1.86,
    'V': 2,
    'XE': 2.16,
    'YB': 2,
    'Y': 2,
    'ZN': 1.29,
    'ZR': 2,
    'X': 1,
}

# Atomic covalent radii dictionary (in upper case!)
atomic_covalent_radius = {
    'AL': 1.21,
    'SB': 1.39,
    'AR': 1.51,
    'AS': 1.21,
    'BA': 2.15,
    'BE': 0.96,
    'BI': 1.48,
    'B': 0.83,
    'BR': 1.21,
    'CD': 1.54,
    'CS': 2.44,
    'CA': 1.76,
    'C': 0.68,
    'CE': 2.04,
    'CL': 0.99,
    'CR': 1.39,
    'CO': 1.26,
    'CU': 1.32,
    'DY': 1.92,
    'ER': 1.89,
    'EU': 1.98,
    'F': 0.64,
    'GD': 1.96,
    'GA': 1.22,
    'GE': 1.17,
    'AU': 1.36,
    'HF': 1.75,
    'HE': 1.5,
    'HO': 1.92,
    'H': 0.23,
    'IN': 1.42,
    'I': 1.4,
    'IR': 1.41,
    'FE': 1.52,
    'KR': 1.5,
    'LA': 2.07,
    'PB': 1.46,
    'LI': 1.28,
    'LU': 1.87,
    'MG': 1.41,
    'MN': 1.61,
    'HG': 1.32,
    'MO': 1.54,
    'ND': 2.01,
    'NE': 1.5,
    'NI': 1.24,
    'NB': 1.64,
    'N': 0.68,
    'OS': 1.44,
    'O': 0.68,
    'PD': 1.39,
    'P': 1.05,
    'PT': 1.36,
    'K': 2.03,
    'PR': 2.03,
    'PA': 2,
    'RE': 1.51,
    'RH': 1.42,
    'RB': 2.2,
    'RU': 1.46,
    'SM': 1.98,
    'SC': 1.7,
    'SE': 1.22,
    'SI': 1.2,
    'AG': 1.45,
    'NA': 1.66,
    'SR': 1.95,
    'S': 1.02,
    'TA': 1.7,
    'TE': 1.47,
    'TB': 1.94,
    'TL': 1.45,
    'TH': 2.06,
    'TM': 1.9,
    'SN': 1.39,
    'TI': 1.6,
    'W': 1.62,
    'U': 1.96,
    'V': 1.53,
    'XE': 1.5,
    'YB': 1.87,
    'Y': 1.9,
    'ZN': 1.22,
    'ZR': 1.75,
    'X': 1,
}

# Atom symbols - atom keys pairs for deciphering OPLS force field
# from dl_field.atom_type file - DL_FIELD 3.5 by C. W. Yong
opls_atom_keys = {
    # The list has to be case sensitive.
    # The OPLS atom types can overlap in case of noble gasses.
    # 'He' - helim, 'HE' or 'he' - hydrogen
    # 'Ne'- neon, 'NE' or 'ne' - imine nitrogen
    # 'Na' - sodium, 'NA' - nitrogen
    # In case of these there still a warning sould be raised
    # as 'ne' can be just lower case for 'Ne'.
    'Ar': ['AR', 'Ar', 'ar'],
    'B': ['B', 'b'],
    'Br': ['BR', 'BR-', 'Br', 'br', 'br-'],
    'C': [
        'CTD', 'CZN', 'C', 'CBO', 'CZB', 'CDS', 'CALK', 'CG', 'CML', 'C5B',
        'CTP', 'CTF', 'C5BC', 'CZA', 'CTS', 'CO', 'C5X', 'CQ', 'CP1', 'CDXR',
        'CANI', 'CRA', 'C4T', 'CHZ', 'CAO', 'CTA', 'CDX', 'CA5', 'CTJ', 'CZ',
        'CO4', 'CTI', 'C5BB', 'CG1', 'C5M', 'CTM', 'CT', 'C5A', 'CN', 'C3M',
        'CB', 'CT1', 'C5N', 'CO3', 'CTQ', 'CTH', 'CTU', 'CTE', 'CTC', 'CTG',
        'C3T', 'CD', 'CME', 'CT_F', 'CA', 'C56B', 'CT1G', 'C56A', 'CM', 'CTNC',
        'CR3', 'ctd', 'czn', 'c', 'cbo', 'czb', 'cds', 'calk', 'cg', 'cml',
        'c5b', 'ctp', 'ctf', 'c5bc', 'cza', 'cts', 'co', 'c5x', 'cq', 'cp1',
        'cdxr', 'cani', 'cra', 'c4t', 'chz', 'cao', 'cta', 'cdx', 'ca5', 'ctj',
        'cz', 'co4', 'cti', 'c5bb', 'cg1', 'c5m', 'ctm', 'ct', 'c5a', 'cn',
        'c3m', 'cb', 'ct1', 'c5n', 'co3', 'ctq', 'cth', 'ctu', 'cte', 'ctc',
        'ctg', 'c3t', 'cd', 'cme', 'ct_f', 'ca', 'c56b', 'ct1g', 'c56a', 'cm',
        'ctnc', 'cr3'
    ],
    'Cl': ['CL', 'CL-', 'Cl', 'cl', 'cl-'],
    'F': [
        'F', 'FX1', 'FX2', 'FX3', 'FX4', 'FG', 'F-', 'f', 'fx1', 'fx2', 'fx3',
        'fx4', 'fg', 'f-'
    ],
    'H': [
        'HA', 'HAE', 'HS', 'HT3', 'HC', 'HWS', 'H', 'HNP', 'HAM', 'H_OH', 'HP',
        'HT4', 'HG', 'HMET', 'HO', 'HANI', 'HY', 'HCG', 'HE', 'ha', 'hae',
        'hs', 'ht3', 'hc', 'hws', 'h', 'hnp', 'ham', 'h_oh', 'hp', 'ht4', 'hg',
        'hmet', 'ho', 'hani', 'hy', 'hcg'
    ],
    'He': ['He'],
    'I': ['I', 'I-', 'i', 'i-'],
    'Kr': ['Kr', 'kr'],
    'N': [
        'NAP', 'NN', 'NB', 'N5BB', 'NS', 'NOM', 'NTC', 'NP', 'N', 'NTH2',
        'NTH', 'NZC', 'NO', 'N5B', 'NO3', 'NZT', 'NZ', 'NI', 'NTH0', 'NA5B',
        'NT', 'NO2', 'NBQ', 'NG', 'NE', 'NZA', 'NA', 'NZB', 'NHZ', 'NO2B',
        'NEA', 'NA5', 'NE', 'nap', 'nn', 'nb', 'n5bb', 'ns', 'nom', 'ntc',
        'np', 'n', 'nth2', 'nth', 'nzc', 'no', 'n5b', 'no3', 'nzt', 'nz', 'ni',
        'nth0', 'na5b', 'nt', 'no2', 'nbq', 'ng', 'nza', 'nzb', 'nhz', 'no2b',
        'nea', 'na5'
    ],
    'Na': ['Na', 'Na+'],
    'Ne': ['Ne'],
    'O': [
        'OM', 'OAB', 'ONI', 'O2ZP', 'O2Z', 'OHE', 'OES', 'OBS', 'OT4', 'OWS',
        'O3T', 'OT3', 'O4T', 'OAL', 'O2', 'OAS', 'OS', 'ON', 'OVE', 'OZ', 'O',
        'OHX', 'OY', 'ONA', 'OA', 'OHP', 'OSP', 'OH', 'om', 'oab', 'oni',
        'o2zp', 'o2z', 'ohe', 'oes', 'obs', 'ot4', 'ows', 'o3t', 'ot3', 'o4t',
        'oal', 'o2', 'oas', 'os', 'on', 'ove', 'oz', 'o', 'ohx', 'oy', 'ona',
        'oa', 'ohp', 'osp', 'oh'
    ],
    'P':
        ['P', 'P1', 'P2', 'P3', 'P4', 'PR', 'p', 'p1', 'p2', 'p3', 'p4', 'pr'],
    'Rn': ['Rn', 'rn'],
    'S': [
        'S', 'SX6', 'SY', 'SH', 'SA', 'SZ', 'SD', 's', 'sx6', 'sy', 'sh', 'sa',
        'sz', 'sd'
    ],
    'Xe': ['Xe', 'xe']
}
