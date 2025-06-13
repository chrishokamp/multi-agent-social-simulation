import { Person } from "./Person.js";
import { utils } from "./utils.js";
import { GameObject } from "./GameObject.js";
import { OverworldEvent } from "./OverworldEvent.js";
import { NPC } from "./NPC.js";
// import { loadConfigFromFile } from "vite";

// Files in public directory are served on root path
const placeholderImages = '/placeholderImages'

export class OverworldMap {
    constructor(config) {
        this.gameObjects = config.gameObjects;
        this.walls = config.walls || {};

        this.lowerImage = new Image();
        this.lowerImage.src = config.lowerSrc;
        this.lowerImageLoaded = false;
        this.lowerImage.onload = () => {
            this.lowerImageLoaded = true;
            console.log(`Loaded image: ${this.lowerImage.src}`);
        };
        this.lowerImage.onerror = () => {
            console.error(`Failed to load image: ${config.lowerSrc}`);
        };

        this.upperImage = new Image();
        this.upperImage.src = config.upperSrc;
        this.upperImageLoaded = false;
        this.upperImage.onload = () => {
            this.upperImageLoaded = true;
            console.log(`Loaded image: ${this.upperImage.src}`);
        };
        this.upperImage.onerror = () => {
            console.error(`Failed to load image: ${config.upperSrc}`);
        };

        this.isCutscenePlaying = false;
    }

    drawLowerImage(ctx, cameraPerson) {
        if (this.lowerImageLoaded) {
            //console.log("Drawing lower image...");
            ctx.drawImage(
                this.lowerImage,
                0,//utils.withGrid(10.5) - cameraPerson.x,
                0,//utils.withGrid(6) - cameraPerson.y
            )
        }
    }

    drawUpperImage(ctx, cameraPerson) {
        if (this.upperImageLoaded) {
            //console.log("Drawing upper image...");
            ctx.drawImage(
                this.upperImage,
                0,//utils.withGrid(10.5) - cameraPerson.x,
                0,//utils.withGrid(6) - cameraPerson.y
            )
        }
    }

    isSpaceTaken(currentX, currentY, direction) {
        const { x, y } = utils.nextPosition(currentX, currentY, direction);
        return this.walls[`${x},${y}`] || false;
    }

    mountObjects() {
        Object.keys(this.gameObjects).forEach(key => {

            let object = this.gameObjects[key];
            object.id = key;

            object.mount(this);
        })
    }

    unmountObjects() {
        Object.keys(this.gameObjects).forEach(key => {
            let object = this.gameObjects[key];
            if (object.unmount) {
                object.unmount(); // Ensure each object can clean up if necessary
            }
        });
        this.gameObjects = {}; // Clear all game objects
    }

    async startCutscene(events) {
        this.isCutscenePlaying = true;

        for (let i = 0; i < events.length; i++) {
            const eventHandler = new OverworldEvent({
                event: events[i],
                map: this,
            })
            await eventHandler.init();
        }

        this.isCutscenePlaying = false;
    }

    addWall(x, y) {
        this.walls[`${x},${y}`] = true;
    }
    removeWall(x, y) {
        delete this.walls[`${x},${y}`]
    }
    moveWall(wasX, wasY, direction) {
        console.log(`Moving wall from (${wasX}, ${wasY}) in direction: ${direction}`);
        this.removeWall(wasX, wasY);
        const { x, y } = utils.nextPosition(wasX, wasY, direction);
        this.addWall(x, y);
    }

    destroy() {
        // Reset walls
        this.walls = {};

        // Unmount all game objects
        this.unmountObjects();

        // Reset map state
        this.lowerImageLoaded = false;
        this.upperImageLoaded = false;

        console.log("OverworldMap instance destroyed.");
    }

}

window.OverworldMaps = {
    DemoRoom: {
        lowerSrc: `${placeholderImages}/maps/DemoLower.png`,
        upperSrc: `${placeholderImages}/maps/DemoUpper.png`,
        gameObjects: {
            hero: new Person({
                agentType: "agent3",
                isPlayerControlled: true,
                x: utils.withGrid(7),
                y: utils.withGrid(4),
                src: `${placeholderImages}/characters/people/hero.png`
            }),
            salesman: new Person({
                agentType: "agent2",
                x: utils.withGrid(6),
                y: utils.withGrid(7),
                npcConfig: {
                    dialogueQueue: [{ text: "test" }],
                    // other NPC-specific configurations can go here
                },
                src: `${placeholderImages}/characters/people/npc1.png`,
                behaviourLoop: [
                    { type: "stand", direction: "left", time: 3200 },
                    { type: "stand", direction: "up", time: 3200 },
                    { type: "stand", direction: "right", time: 4800 },
                    { type: "stand", direction: "up", time: 1200 },
                ]

            }),
            customer: new Person({
                agentType: "agent1",
                x: utils.withGrid(4),
                y: utils.withGrid(7),
                npcConfig: {
                    dialogueQueue: [{ text: "test" }],
                    // other NPC-specific configurations can go here
                },
                src: `${placeholderImages}/characters/people/npc3.png`,
                behaviourLoop: [
                    { type: "stand", direction: "up", time: 3200 },
                    { type: "stand", direction: "right", time: 3200 },
                    { type: "stand", direction: "up", time: 4800 },
                    { type: "stand", direction: "left", time: 1200 },
                ]
            }),
        },
        walls: {
            [utils.asGridCoord(7, 6)]: true, [utils.asGridCoord(1, 3)]: true,
            [utils.asGridCoord(8, 6)]: true, [utils.asGridCoord(2, 3)]: true,
            [utils.asGridCoord(7, 7)]: true, [utils.asGridCoord(3, 4)]: true,
            [utils.asGridCoord(8, 7)]: true, [utils.asGridCoord(4, 4)]: true,
            [utils.asGridCoord(5, 3)]: true, [utils.asGridCoord(8, 10)]: true,
            [utils.asGridCoord(6, 4)]: true, [utils.asGridCoord(7, 10)]: true,
            [utils.asGridCoord(7, 3)]: true, [utils.asGridCoord(6, 10)]: true,
            [utils.asGridCoord(8, 4)]: true, [utils.asGridCoord(5, 11)]: true,
            [utils.asGridCoord(9, 3)]: true, [utils.asGridCoord(4, 10)]: true,
            [utils.asGridCoord(10, 3)]: true, [utils.asGridCoord(3, 10)]: true,
            [utils.asGridCoord(11, 4)]: true, [utils.asGridCoord(2, 10)]: true,
            [utils.asGridCoord(11, 5)]: true, [utils.asGridCoord(1, 10)]: true,
            [utils.asGridCoord(11, 6)]: true, [utils.asGridCoord(0, 9)]: true,
            [utils.asGridCoord(11, 7)]: true, [utils.asGridCoord(0, 8)]: true,
            [utils.asGridCoord(11, 8)]: true, [utils.asGridCoord(0, 7)]: true,
            [utils.asGridCoord(11, 9)]: true, [utils.asGridCoord(0, 6)]: true,
            [utils.asGridCoord(10, 10)]: true, [utils.asGridCoord(0, 5)]: true,
            [utils.asGridCoord(9, 10)]: true, [utils.asGridCoord(0, 4)]: true,
        }
    },
    Kitchen: {
        lowerSrc: `${placeholderImages}/maps/KitchenLower.png`,
        upperSrc: `${placeholderImages}/maps/KitchenUpper.png`,
        gameObjects: {
            hero: new GameObject({
                x: 3,
                y: 5,
            }),
            npcA: new GameObject({
                x: 9,
                y: 6,
                src: `${placeholderImages}/characters/people/npc2.png`
            }),
            npcB: new NPC({
                x: 10,
                y: 8,
                src: `${placeholderImages}/characters/people/npc3.png`,
                npcConfig: {
                    dialogueQueue: [{ text: "test" }]
                }
            })
        }
    },
}