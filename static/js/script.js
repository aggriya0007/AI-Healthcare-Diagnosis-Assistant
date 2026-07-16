function searchSymptoms(){

    let input = document.getElementById("searchBox").value.toLowerCase();

    let symptoms = document.getElementsByClassName("symptom-item");

    for(let i=0;i<symptoms.length;i++){

        let text = symptoms[i].innerText.toLowerCase();

        if(text.includes(input)){

            symptoms[i].style.display="block";

        }

        else{

            symptoms[i].style.display="none";

        }

    }

}