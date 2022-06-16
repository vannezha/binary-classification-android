package com.example.isenglish

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import com.example.isenglish.ml.LanguageClassification
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.json.JSONArray
import org.json.JSONObject
import java.io.InputStream

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

//        create IO UI
        val tv : TextView = findViewById(R.id.textView)
        val button : Button = findViewById(R.id.button)

        button.setOnClickListener {
    //            get input (string)
            val ed: EditText = findViewById(R.id.inputIsEnglish)
            val inputIsEnglish: String = ed.text.toString()
            val isEnglish = isEnglish(inputIsEnglish)
    //          show it in textView
            tv.text = isEnglish.toString()
        }
    }

    //    5 function that needed to do input preprossing
    private fun isEnglish(inputIsEnglish: String): Boolean {
        //            alocate some memory
        val inputFeature = TensorBuffer.createFixedSize(intArrayOf(1, 32), DataType.FLOAT32)
        //          preprocess input
        val seq = this.sequenize(this.splitLowerRemovePunctuation(inputIsEnglish))
        //          initiate a model
        val model = LanguageClassification.newInstance(this)
        //          doing inference and close the model after it
        inputFeature.loadArray(seq.toFloatArray())
        val outputs = model.process(inputFeature)
        model.close()
        //          output (boolean)
        val outputFeature = outputs.outputFeature0AsTensorBuffer.floatArray
        return outputFeature[0] < 0.5
    }
    private fun wordIndex(): Map<String, *> {
        val json: String?
        val inputStream: InputStream = assets.open("IsEnglish.json")
        json = inputStream.bufferedReader().use { it.readText() }.toString()
        return JSONObject(json).toMap()
    }
    private fun JSONObject.toMap(): Map<String, *> = keys().asSequence().associateWith { it ->
        when (val value = this[it])
        {
            is JSONArray ->
            {
                val map = (0 until value.length()).associate { Pair(it.toString(), value[it]) }
                JSONObject(map).toMap().values.toList()
            }
            is JSONObject -> value.toMap()
            JSONObject.NULL -> null
            else            -> value
        }
    }
    private fun splitLowerRemovePunctuation(source:String): Array<String> {
        val str = source.replace("[!\"#$%&'()*+,-./:;<=>?@\\[\\]^_`{|}~]".toRegex(), "").trim()
        return str.split("\\s+".toRegex()).toTypedArray()
    }
    private fun sequenize(source: Array<String>):ArrayList<Float> {
        var counter = 32
        val seq = arrayListOf<Float>()
        for (i in source){
            if(i in wordIndex().keys){
                if (counter == 0){
                    return seq
                }else{
                    counter -= 1
                    seq.add(wordIndex().get(i).toString().toFloat())
                }
            }else{
                if (counter == 0){
                    return seq
                }else{
                    counter -= 1
                    seq.add(1f)
                }
            }
        }
        for (i in 1..counter){
            seq.add(0f)
        };return seq
    }
}



