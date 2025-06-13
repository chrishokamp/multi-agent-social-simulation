
export class OverworldEvent {
    constructor({ map, event }) {
        this.map = map;
        this.event = event;
    }

     // Helper to remove event listeners
    removeEventListener(type, handler) {
        document.removeEventListener(type, handler);
    }

    stand(resolve) {
        const who = this.map.gameObjects[this.event.who];
        who.startBehaviour({
            map: this.map
        }, {
            type: "stand",
            direction: this.event.direction,
            time: this.event.time
        })

        //Set up a handler to complete when correct person is done waiting, then resolve the event
        const completeHandler = e => {
            if(e.detail.whoId === this.event.who) {
                document.removeEventListener("PersonStandComplete", completeHandler);
                resolve();
            }
        }
        document.addEventListener("PersonStandComplete", completeHandler)
    }

    walk(resolve) {
        const who = this.map.gameObjects[ this.event.who ];
        if (!who) {
            console.error("Invalid gameObject for 'who':", this.event.who, this.map.gameObjects);
            throw new Error(`GameObject not found for 'who': ${this.event.who}`);
        }
        who.startBehaviour({
            map: this.map
        }, {
            type: "walk",
            direction: this.event.direction,
            retry: true
        })

        //Set up a handler to complete when correct person is done walking, then resolve the event
        const completeHandler = e => {
            if(e.detail.whoId === this.event.who) {
                document.removeEventListener("PersonWalkingComplete", completeHandler);
                resolve();
            }
        }

        document.addEventListener("PersonWalkingComplete", completeHandler)

    }

    init() {
        return new Promise(resolve => {
            this[this.event.type](resolve)
        })
    }

}