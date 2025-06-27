import React from 'react';
import { navLinks } from '../../constants/index';

const Navbar = () => {
  const NavItems = () => {
    return (
      <ul className="flex gap-6">
        {navLinks.map(({ id, href, name, icon: Icon }) => (
          <li key={id} className="nav-li">
            <a
              href={href}
              className="flex items-center gap-2 text-md"
              style={{ color: 'hsl(var(--oncolor-100))' }}
              onMouseOver={e => e.currentTarget.style.color = 'hsl(var(--bg-200))'}
              onMouseOut={e => e.currentTarget.style.color = 'hsl(var(--oncolor-100))'}
            >
              <Icon
                className={name === 'Dashboard' ? 'h-6 w-6 -mr-0.75' : 'h-5 w-5'}
                style={{ color: 'hsl(var(--accent-pro-200))' }}
              />
              {name}
            </a>
          </li>
        ))}
      </ul>
    );
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-[60]">
      <div
        style={{
          background: 'hsl(var(--accent-main-000))'
        }}
      >
        <div className="max-w-7xl mx-auto flex justify-between items-center py-6">
          <div className="flex items-center gap-2 text-md">
            <a href="/" className="navbar-brand flex items-center ">
              NegotiationGym
            </a>
          </div>
          <nav className="sm:flex hidden">
            <NavItems />
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
