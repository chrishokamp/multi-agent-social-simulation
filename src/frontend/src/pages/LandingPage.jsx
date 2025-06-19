import React from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import Tilt from 'react-parallax-tilt';

import HeroScene3D from './components/HeroScene3D';
import Navbar from './components/Navbar';
import Footer from './components/Footer';

import { features } from '../constants/index';

const LandingPage = () => {
  return (
    <main className="relative">
      <Navbar />
      <section className="relative w-full h-screen z-10">
        <div className="absolute inset-0">
          <Canvas camera={{ position: [0, 2, 8], fov: 50 }}>
            <OrbitControls enableZoom={false} />
            <ambientLight intensity={1} />
            <directionalLight position={[0, 10, 5]} intensity={1} />
            <HeroScene3D url="/models/robot_playground.glb" />
          </Canvas>
        </div>
      </section>
      <section className="py-12 -mt-22 mb-20">
        <div className="container mx-auto px-4">
          <div className="mb-8 text-center">
            <h2 className="text-3xl text-white font-bold mb-2">Key Features</h2>
            <p className="text-gray-300 text-lg max-w-xl mx-auto">
              Our framework makes it easy to create and analyze complex multi-agent simulations
              powered by LLMs.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {features.map((feature) => (
              <Tilt>
                <div
                  key={feature.title}
                  className="rounded-2xl border border-violet-400 shadow p-6 hover:shadow-md transition"
                >
                  <div className="flex items-center mb-4">
                    {/* Icon */}
                    <feature.icon className="h-8 w-8 text-violet-500 mr-3" />
                    <h3 className="text-xl text-white font-semibold">{feature.title}</h3>
                  </div>
                  <p className="text-violet-300">{feature.description}</p>
                </div>
              </Tilt>
            ))}
          </div>
          <div className="container mx-auto mt-8 px-4 text-center">
            <p className="text-lg text-gray-300 mb-4">
              Like what you see? Start building your simulations now.
            </p>
            <a
              href="/configurator"
              className="inline-block px-8 py-3 bg-violet-800 text-white font-semibold rounded-3xl hover:shadow-button transition"
            >
              Get Started
            </a>
          </div>
        </div>
      </section>
      <Footer />
    </main>
  );
};

export default LandingPage;
