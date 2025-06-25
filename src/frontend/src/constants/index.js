import React from 'react';

import { GrConfigure } from "react-icons/gr";
import { FaListUl } from "react-icons/fa6";
import { FaImage } from 'react-icons/fa6';
import { RiDashboardHorizontalFill } from "react-icons/ri";

export const navLinks = [
    {
        id: 1,
        name: 'Configurator',
        href: '/configurator',
        icon: GrConfigure,
    },
    {
        id: 2,
        name: 'Catalog',
        href: '/simulations',
        icon: FaListUl,
    },
    {
        id: 4,
        name: 'Dashboard',
        href: '/dashboard',
        icon: RiDashboardHorizontalFill,
    },
];

export const features = [
    {
        title: 'Agent Configurator',
        description:
            'Define roles, goals, and hyperparameters. Queue your simulation. Let us do the rest.',
        icon: GrConfigure,
    },
    {
        title: 'Analytics Dashboard',
        description:
            'Track performance metrics, outcome distributions, and refine your approach with continuous feedback.',
        icon: RiDashboardHorizontalFill,
    },
];