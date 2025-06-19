import { util } from "echarts";
import { GameObject } from"./GameObject.js";
import { utils } from "./utils.js";
import { NPC } from "./NPC.js";
import { Sprite } from "./Sprite.js";

export class Person extends GameObject {
    constructor(config) {
        super(config);
        this.movingProgressRemaining = 0;
        this.isSpeechBubble = config.speechBubble || false;
        this.agentType = config.agentType || null; 
        this.isTalking = false;

        this.isPlayerControlled = config.isPlayerControlled || false;

        if (!this.isSpeechBubble) {
            this.speechBubble = new Person({
                speechBubble: true,
                x: utils.withGrid(this.x + 0.5),
                y: utils.withGrid(this.y - 0.5),
                src: "placeholderImages/characters/speechBubble.png",
            });
        }

        this.directionUpdate = {
            "up": ["y", -1],
            "down": ["y", 1],
            "left": ["x", -1],
            "right": ["x", 1],
        };
    }

    update(state) { 
        super.update();

        if(this.movingProgressRemaining > 0) {
            this.updatePosition();
        } else {

            //More cases for starting to walk will come here
            //
            //

            //Case: We're keyboard ready and have an arrow pressed
            if(!state.map.isCutscenePlaying && this.isPlayerControlled && state.arrow) {
                this.startBehaviour(state, {
                    type: "walk",
                    direction: state.arrow
                })
            }
            this.updateSprite(state);
        }
        //console.log("window.agentSpeaking: ", window.agentSpeaking);

        if('agent'+ window.agentSpeaking == this.agentType && state.ctx) {
            //console.log("I am speaking: ", this.agentType);
            this.drawSpeechBubble(state.ctx); 
        }
        
    }

    startBehaviour(state, behaviour) {
        //Set character direction to whatever behaviour has
        this.direction = behaviour.direction;

        if(behaviour.type === "walk") {

            if (!state.map) {
                console.error("Error: state.map is undefined during startBehaviour");
                return;
            }

            //Stop here if space is not free
            if(state.map.isSpaceTaken(this.x, this.y, this.direction)) {
                behaviour.retry && setTimeout(() => {
                    this.startBehaviour(state, behaviour)
                }, 10);

                return;
            }

            //Ready to walk!
            state.map.moveWall(this.x, this.y, this.direction);
            this.movingProgressRemaining = 16;
            this.updateSprite(state);
            console.log("Movement started.");
        }

        if(behaviour.type === "stand") {
            setTimeout(() => {
                utils.emitEvent("PersonStandComplete", {
                    whoId: this.id
                })
            }, behaviour.time)
        }

    }

    updatePosition() {
        const [property, change] = this.directionUpdate[this.direction];
        this[property] += change;
        this.movingProgressRemaining -= 1;

        if(this.movingProgressRemaining === 0) {
            // Only emit if the person is still mounted
        if (this.isMounted) {
            utils.emitEvent("PersonWalkingComplete", {
                whoId: this.id
            });
        } else {
            console.warn(`Person ${this.id} unmounted; skipping walking complete event.`);
        }


        }
        
    }

    updateSprite() {

        if(this.movingProgressRemaining > 0) {
            this.sprite.setAnimation("walk-"+ this.direction);
            return;
        }
        this.sprite.setAnimation("idle-"+this.direction);
    }
/*
    startSpeech(){
        //console.log("start speech for ", this.agentType);
        this.isTalking = true;

    }
    
    stopSpeech(){
        //console.log("stop speech for ", this.agentType);
        this.isTalking = false;
    }*/

    drawSpeechBubble(ctx) {
        //console.log("draw speech bubble");
        // Draw the speech bubble
        const bubbleHeight = 10; // Fixed height for the bubble
            
        const speechText = "...";
        const bubbleWidth = speechText.length + 15;


        const x = this.x+5; // Calculate position based on NPC's position
        const y = this.y; // Position above the NPC

        ctx.fillStyle = "white";
        ctx.fillRect(x - bubbleWidth / 2, y-20, bubbleWidth, bubbleHeight); // Background of the bubble

        ctx.strokeStyle = "black";
        ctx.lineWidth = 1;
        ctx.strokeRect(x - bubbleWidth / 2, y-20, bubbleWidth, bubbleHeight); // Border of the bubble

        // Draw the text inside the bubble
        ctx.fillStyle = "black";
        ctx.font = "12px Arial";
        ctx.fillText(speechText, x - bubbleWidth / 4, y + bubbleHeight / 1.5 - 20);
    }  

}