import express from "express";
import { handleChat } from "../controller/chat.controler.js";

const chatRouter = express.Router();

chatRouter.post("/chat", handleChat);

export default chatRouter;
