import { Sprite } from "./Sprite.js";
import { OverworldEvent } from "./OverworldEvent.js";

export class GameObject{
    constructor(config) {
        this.id = config.id || `gameObject_${Math.random().toString(36).substr(2, 9)}`;
        this.isMounted = false;
        this.ctx=null
        this.x = config.x || 0;
        this.y = config.y || 0;
        this.direction = config.direction || "down";
        this.sprite = new Sprite({
            gameObject: this,
            src: config.src || "placeholderImages/characters/people/npc2.png",
        });

        this.behaviourLoop = config.behaviourLoop || [];
        this.behaviourLoopIndex = 0;
    }

    mount(map) {
        //console.log("mounting!")
        this.isMounted = true;
        map.gameObjects[this.id] = this; // Register the GameObject in the map
        map.addWall(this.x, this.y);

        //If we have a behaviour, kick off after a short delay
        setTimeout(() => {
            this.doBehaviourEvent(map);
        }, 10)

    }

    unmount(map) {
        //console.log("Unmounting GameObject:", this.id);
        this.isMounted = false;
        if (map) {
            map.removeWall(this.x, this.y);
        }
        // Reset any active behaviors or states
        this.behaviourLoopIndex = 0; // Reset the behavior loop
    }

    resetState() {
        //console.log("Resetting GameObject state:", this.id);
        this.x = this.initialX;
        this.y = this.initialY;
        this.direction = "down"; // Reset to default direction
        this.behaviourLoopIndex = 0; // Reset behavior loop
    }

    update() {

    }

    async doBehaviourEvent(map) {

        //Don't do anything if there is a more important cutscene or I don't have config to do anything anyway.
        if(map.isCutscenePlaying || this.behaviourLoop.length === 0) {
            return;
        }

        //Setting up our event with relevant info
        const eventConfig = this.behaviourLoop[this.behaviourLoopIndex];
        if (!eventConfig) {
            console.error("Invalid behaviourLoop configuration:", this.behaviourLoop);
        return;
        }
        eventConfig.who = this.id;

        //Create an event instance out of our next event config
        const eventHandler = new OverworldEvent({ map, event: eventConfig});
        await eventHandler.init();

        //Setting the next event to fire
        this.behaviourLoopIndex += 1;
        if(this.behaviourLoopIndex === this.behaviourLoop.length) {
            this.behaviourLoopIndex = 0;
        }

        //Do it again!
        this.doBehaviourEvent(map);

    }
}