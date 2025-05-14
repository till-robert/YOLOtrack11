
window.addEventListener('load', () => {
    console.log(document.querySelectorAll('.reference.internal.image-reference > img'));
    el = document.querySelectorAll('.reference.internal.image-reference')

    el.forEach(el => {
        el.style.width = "100%";
        img = el.children[0];
        img.style.maxHeight = img.style.height;
        img.style.maxWidth = img.style.width;
        img.style.width = "inherit";
        img.style.removeProperty('height');
    });
    console.log('Custom JS loaded: Removed width and height attributes from internal image references.');
});