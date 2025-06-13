import { OverworldMap } from "./OverworldMap.js";
import { DirectionInput } from "./DirectionInput.js";;

export class Overworld {
    constructor(config) {
        this.element = config.element;
        this.canvas = this.element.querySelector(".game-canvas");
        this.ctx = this.canvas.getContext("2d");
        this.map = null;
        this.lastTime = 0;
        this.isRunning = false; //Add flag to control the game loop
    }

    startGameLoop() {
        this.isRunning = true;
        const step = (timestamp) => {
            if (!this.isRunning) return; // Stop the game loop if it's no longer running

            const deltaTime = timestamp - this.lastTime; // Time difference between frames
            this.lastTime = timestamp;

            //Clear off the canvas
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

            //Establish the camera person
            const cameraPerson = this.map.gameObjects.hero;

            //Draw Lower layer
            this.map.drawLowerImage(this.ctx, cameraPerson);

            //Draw Game Objects
            Object.values(this.map.gameObjects).sort((a, b) => {
                return a.y - b.y;
            }).forEach(object => {
                object.sprite.draw(this.ctx, cameraPerson);
            })

            //Draw Upper layer
            this.map.drawUpperImage(this.ctx, cameraPerson);

            

            //Update all objects
            Object.values(this.map.gameObjects).forEach(object => {
                
                if (!window.isSimPaused) { // Only update when playback is is not paused
                  object.update({
                    arrow: this.directionInput.direction,
                    map: this.map,
                    ctx: this.ctx,
                    deltaTime: deltaTime / 1000 // Convert to seconds
                  });


                }
              });

            requestAnimationFrame(() => {
                step();
            })
        }
        step();
    }

    init() {
        this.map = new OverworldMap(window.OverworldMaps.DemoRoom);
        this.map.mountObjects();

        this.directionInput = new DirectionInput();
        this.directionInput.init();
  
        this.startGameLoop();

        // this.map.startCutscene([
        //     { who: "hero", type: "walk", direction: "down" },
        //     { who: "hero", type: "walk", direction: "down" },
        //     { who: "salesman", type: "walk", direction: "left" },
        //     { who: "salesman", type: "walk", direction: "left" },
        //     { who: "salesman", type: "stand", direction: "up", time: 800 },
        // ])
    }

    destroy() {

        console.log("Destroying Overworld instance...");
        // Stop the game loop
        this.isRunning = false;

        // Clear the canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Reset the map and game objects
        if (this.map) {
            this.map.unmountObjects();
            this.map = null;
        }

        console.log("Overworld instance destroyed and cleaned up.");
    }

}