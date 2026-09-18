import express from "express";
import upload from "../configs/multer.js";
import { splitText } from "../controller/split.controler.js";

const splitRouter = express.Router();

splitRouter.post("/split", upload.single("file"), splitText);

export default splitRouter;
