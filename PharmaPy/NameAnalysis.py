#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:13:10 2020

@author: dcasasor
"""

import numpy as np
from PharmaPy.Gaussians import gaussian


def getBipartite(first, second):
    graph = {}
    types = {}
    comp_names = ['conc', 'frac']
    amount_names = ['mass', 'moles', 'vol']

    num_first = len(first)
    for two in second:
        for count, one in enumerate(first):
            if any(word in one for word in comp_names) and any(word in two for word in comp_names):
                graph[two] = one
                types['composition'] = (one, two)
                break
            elif 'flow' in one and 'flow' in two:
                graph[two] = one
                types['flow'] = (one, two)
                break
            elif 'distr' in one and 'distr' in two:
                graph[two] = one
                types['distrib'] = (one, two)
                break
            elif any(one == word for word in amount_names) and any(two == word for word in amount_names):
                graph[two] = one
                types['amount'] = (one, two)
            elif one == two:
                graph[two] = one
                break
            elif count == num_first - 1:
                graph[two] = None

    return graph, types


def get_types(names):
    conc_type = [name for name in names
                 if ('conc' in name) or ('frac' in name)][0]

    distrib_type = [name if 'distr' in name else None
                    for name in names][0]

    distrib_type = list(filter(lambda x: 'distr' in x, names))

    flow_type = list(filter(lambda x: 'flow' in x, names))

    amount_type = list(
        filter(lambda x: x == 'mass' or x == 'moles' or x == 'vol', names))

    if not distrib_type:
        distrib_type = None
    else:
        distrib_type = distrib_type[0]

    if not flow_type:
        flow_type = None
    else:
        flow_type = flow_type[0]

    if not amount_type:
        amount_type = None
    else:
        amount_type = amount_type[0]

    return conc_type, distrib_type, flow_type, amount_type


def get_dict_states(names, num_species, num_distr, states):
    count = 0
    dict_states = {}

    for name in names:
        if 'conc' in name or 'frac' in name:
            idx_composition = range(count, count + num_species)
            dict_states[name] = states.T[idx_composition].T

            count += num_species
        elif 'distrib' in name:
            idx_distrib = range(count, count + num_distr)
            dict_states[name] = states.T[idx_distrib].T

            count += num_distr
        else:
            dict_states[name] = states.T[count].T

            count += 1

    return dict_states


class NameAnalyzer:
    def __init__(self, names_up, names_down, num_species, num_distr=None):
        self.names_up = names_up
        self.names_down = names_down

        self.num_species = num_species
        self.num_distr = num_distr

        self.bipartite, self.conv_types = getBipartite(names_up, names_down)

    def get_idx(self):
        count = 0

        idx_flow = None
        idx_amount = None
        idx_distrib = None

        for name in self.names_up:
            if 'conc' in name or 'frac' in name:
                idx_composition = range(count, count + self.num_species)

                count += self.num_species
            elif 'distrib' in name:
                idx_distrib = range(count, count + self.num_distr)

                count += self.num_distr
            else:
                if 'flow' in name:
                    idx_flow = count
                elif name == 'mass' or name == 'moles' or name == 'vol':
                    idx_amount = count

                count += 1

        return idx_composition, idx_flow, idx_amount, idx_distrib

    def convertUnits(self, matter_transf):
        y_upstr = matter_transf.y_upstream

        conversion_keys = self.conv_types.keys()
        dict_states = get_dict_states(self.names_up, self.num_species,
                                      self.num_distr, y_upstr)

        y_inlet = y_upstr.copy()

        comp_idx, flow_idx, amount_idx, distrib_idx = self.get_idx()

        # Composition
        comp = self.conv_types['composition']
        if comp[0] != comp[1]:
            state_comp = self.__convertComposition(*comp, dict_states[comp[0]],
                                                   matter_transf)

            dict_states[comp[1]] = state_comp

            y_inlet[:, comp_idx] = state_comp

        if 'flow' in conversion_keys:
            flow = self.conv_types['flow']
            if flow[0] != flow[1]:
                state_flow = self.__convertFlow(*flow, dict_states[flow[0]],
                                                matter_transf,
                                                dict_states[comp[0]],
                                                comp[0])

                dict_states[flow[1]] = state_flow

                y_inlet[:, flow_idx] = state_flow

        elif 'amount' in conversion_keys:
            amount = self.conv_types['amount']
            if amount[0] != amount[1]:
                state_amount = self.__convertFlow(*amount,
                                                  dict_states[amount[0]],
                                                  matter_transf)

                dict_states[amount[1]] = state_amount

                y_inlet[:, amount_idx] = state_amount

        if 'distrib' in conversion_keys:
            distr = self.conv_types['distrib']
            if distr[0] != distr[1]:
                distrib_conv = self.__convert_distrib(*distr,
                                                      dict_states[distr[0]],
                                                      matter_transf)

        matter_transf.y_inlet = y_inlet

        return dict_states

    def __convertComposition(self, prefix_up, prefix_down, composition,
                             matter_object):
        up, down = prefix_up, prefix_down

        if 'frac' in up and 'frac' in down:
            method = getattr(matter_object, 'frac_to_frac')
            if 'mole' in up:
                fun_kwargs = {'mole_frac': composition}
            elif 'mass' in up:
                fun_kwargs = {'mass_frac': composition}

        elif 'frac' in up and 'conc' in down:
            method = getattr(matter_object, 'frac_to_conc')

            if 'mole' in up:
                fun_kwargs = {'mole_frac': composition}
            elif 'mass' in up:
                fun_kwargs = {'mass_frac': composition}

            if 'mass' in down:
                fun_kwargs['basis'] = 'mass'

        elif 'mole_conc' in up and 'frac' in down:
            method = getattr(matter_object, 'conc_to_frac')
            fun_kwargs = {'conc': composition}

            if 'mass' in down:
                fun_kwargs['basis'] = 'mass'

        elif 'mass_conc' in up and 'frac' in down:
            method = getattr(matter_object, 'mass_conc_to_frac')
            fun_kwargs = {'conc': composition}

            if 'mole' in down:
                fun_kwargs['basis'] = 'mole'

        elif 'conc' in up and 'conc' in down:
            method = getattr(matter_object, 'conc_to_conc')

            if 'mole' in up:
                fun_kwargs = {'mole_conc': composition}
            else:
                fun_kwargs = {'mass_conc': composition}

        output_composition = method(**fun_kwargs)

        return output_composition

    def __convertFlow(self, prefix_up, prefix_down, flow, matter_object,
                      composition, comp_name):
        up, down = prefix_up, prefix_down

        # Molecular weight
        if np.asarray(flow).ndim == 0:
            mw_av = matter_object.mw_av
        elif comp_name == 'mole_frac':
            mw_av = np.dot(matter_object.mw, composition.T)
        elif comp_name == 'mass_frac':
            mole_frac = matter_object.frac_to_frac(mass_frac=composition)
            mw_av = np.dot(matter_object.mw, mole_frac.T)
        elif comp_name == 'mole_conc':
            mole_frac = (composition.T / composition.sum(axis=1))
            mw_av = np.dot(matter_object.mw, mole_frac)

        # Density
        if np.asarray(flow).ndim == 0:
            density = matter_object.getDensity()
        elif comp_name == 'mole_frac' or comp_name == 'mass_frac':
            density = matter_object.getDensity(**{comp_name: composition})
        elif comp_name == 'mole_conc':
            density = matter_object.getDensity(mole_frac=mole_frac.T)

        # Convert units
        if 'mass' in up and 'mole' in down:
            flow_out = flow / mw_av * 1000  # mol/s
        elif 'mole' in up and 'mass' in down:
            flow_out = flow * mw_av / 1000  # kg/s
        elif 'vol' in down:
            if 'mole' in up:
                density *= 1000 / mw_av

            flow_out = flow / density  # m3/s

        elif 'vol' in up:
            dens = matter_object.getDensity()  # kg/m3
            if 'mole' in down:
                dens *= 1000 / mw_av

            flow_out = flow * dens  # kg/s - mol/s

        return flow_out

    def __convert_distrib(self, prefix_up, prefix_down, distrib,
                          matter_object):
        up, down = prefix_up, prefix_down

        if 'num' in up and 'total' in down:
            distrib_out = distrib
        elif 'num' in up and 'vol' in down:
            pass

        return distrib_out


if __name__ == '__main__':
    names_up = ['temp', 'mole_conc', 'vol_flow']
    names_down = ['mole_frac', 'temp', 'num_distrib', 'mass_flow']

    num_species = 2
    num_distr = 100

    # Data
    states_up = np.array([300, 0.7, 1, 0.001])
    states_down = np.array([0.35, 0.65, 320])

    x_distrib = np.linspace(0, 1000, num_distr)
    distrib = gaussian(x_distrib, 400, 10, 1e10)

    states_down = np.append(states_down, distrib)
    states_down = np.append(states_down, 1)

    # With name analyzer
    analyzer = NameAnalyzer(names_up, names_down, num_species,
                            num_distr)