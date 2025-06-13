import { GameObject } from "./GameObject.js";

export class NPC extends GameObject {
    constructor(config) {
        super(config); // Initialize base GameObject
        this.dialogueQueue = config.npcConfig.dialogueQueue || []; 
        this.isTalking = false;
        this.dialogueDuration = 10000; // Duration each dialogue is shown (in ms)
        this.dialogueTimer = this.dialogueDuration; // Timer to track the time for each dialogue
        this.nextDialogue = null;
    }

    startDialogue() {
        console.log("start npc dialogue");
        this.dialogueQueue.forEach(dialogue => {
            console.log("npc dialog" +dialogue.text);
        });
        this.isTalking = true;
        this.dialogueTimer = this.dialogueDuration; 
        this.nextDialogue = this.dialogueQueue[0];
    }

    advanceDialogue() {
        if (this.dialogueQueue.length > 0) {
            console.log("advance dialogue");
            return this.dialogueQueue.shift(); // Retrieve the next line of dialogue
        } else {
            this.isTalking = false;
        }   
        return null;
    }

    update({ ctx, deltaTime }) {
        super.update(); // Call the base update method

        
        if (this.dialogueTimer <= 0) {
            console.log("times up move to next");
            // Time is up for the current dialogue, move to the next one
            //this.nextDialogue = this.advanceDialogue();
            if (this.nextDialogue) {
                this.dialogueTimer = this.dialogueDuration; // Reset timer for the next dialogue
            }
        }

        this.drawSpeechBubble(ctx); // Pass map.ctx to draw speech bubble

    }


    drawSpeechBubble(ctx) {
        const speechText = this.nextDialogue;
        if (speechText) {
            console.log("speech text: ", speechText.text);

            // Draw the speech bubble
            const bubbleWidth = ctx.measureText(speechText.text).width; // Calculate the width of the bubble
            const bubbleHeight = 10; // Fixed height for the bubble
            
            const x = this.x; // Calculate position based on NPC's position
            const y = this.y; // Position above the NPC
            console.log("x: " + this.x);

            ctx.fillStyle = "white";
            ctx.fillRect(x, y, bubbleWidth, bubbleHeight); // Draw the background of the bubble
            ctx.strokeStyle = "black";
            ctx.strokeRect(x, y, bubbleWidth, bubbleHeight); // Draw the border of the bubble

            //ctx.fillStyle = "black";
            //ctx.font = "15px Arial";
            //ctx.fillText(speechText.text, x , y+10 ); // Draw the dialogue text
        }
    }
}
